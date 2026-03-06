#!/usr/bin/env python3
"""Fix SystemVerilog constructs for iverilog 10.1 compatibility."""

import re
import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_matching_paren(s, start):
    """Find index after matching closing paren for '(' at s[start]."""
    assert s[start] == '('
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    return i


def replace_type_cast(content, type_name, replacement_func):
    """Replace type_name'(expr) with replacement_func(expr_str)."""
    pattern = type_name + "'("
    result = []
    i = 0
    while i < len(content):
        idx = content.find(pattern, i)
        if idx == -1:
            result.append(content[i:])
            break
        if idx > 0 and (content[idx-1].isalnum() or content[idx-1] == '_'):
            result.append(content[i:idx + len(pattern)])
            i = idx + len(pattern)
            continue
        result.append(content[i:idx])
        paren_pos = idx + len(type_name) + 1
        end = find_matching_paren(content, paren_pos)
        expr = content[paren_pos + 1:end - 1].strip()
        result.append(replacement_func(expr))
        i = end
    return ''.join(result)


def fix_enums(content):
    """Replace typedef enum with localparam constants."""
    enum_pattern = re.compile(
        r'  typedef enum logic \[(\d+):0\] \{([^}]+)\} (\w+);',
        re.DOTALL
    )

    # Collect enum type info for later variable fixup
    enum_info = {}

    def enum_replacer(m):
        width = int(m.group(1))
        body = m.group(2)
        type_name = m.group(3)
        values = []
        for line in body.split('\n'):
            v = line.split('//')[0].strip().rstrip(',').strip()
            if v:
                values.append(v)
        enum_info[type_name] = width
        lines = []
        for i, v in enumerate(values):
            lines.append(f'  localparam [{width}:0] {v} = {i};')
        return '\n'.join(lines)

    content = enum_pattern.sub(enum_replacer, content)

    # Now fix variable declarations using these enum types
    for type_name, width in enum_info.items():
        # "type_name var1, var2;" -> "reg [width:0] var1, var2;"
        content = re.sub(
            rf'\b{type_name}\b\s+([\w][\w\s,]*);',
            rf'reg [{width}:0] \1;',
            content
        )

    return content


def fix_for_int_loops(content):
    """Replace for(int var = ...) with pre-declared integer variables.

    iverilog 10.1 segfaults when `int` loop variables are used in
    multiplication expressions within array indices. Also crashes at
    runtime when for(int ...) is used inside package functions.
    Using `integer` variables instead avoids all these issues.

    Strategy: collect all `for (int X` variable names within each
    module/function scope, add `integer X;` declarations, and
    remove the `int` from the for statement.
    """
    # Find all unique variable names used in for(int ...) loops
    var_names = set(re.findall(r'for\s*\(\s*int\s+(\w+)\s*=', content))
    if not var_names:
        return content

    # Replace "for (int X =" with "for (X =" throughout
    content = re.sub(r'for\s*\(\s*int\s+(\w+)\s*=', r'for (\1 =', content)

    # Insert integer declarations at the right scope level.
    decl_line = '  integer ' + ', '.join(sorted(var_names)) + ';\n'

    # For modules: insert after port list closing ');'
    # The port list ends with ');\n' — find the LAST ');' before the module body
    m = re.search(r'^module\s+\w+', content, re.MULTILINE)
    if m:
        # Find ');\n' which closes the port list (not just any ';')
        port_close = re.search(r'\);\s*\n', content[m.start():])
        if port_close:
            insert_pos = m.start() + port_close.end()
            content = content[:insert_pos] + decl_line + content[insert_pos:]

    # For functions: insert integer decl after the function signature line
    # Pattern: "function ... ;\n" -> insert "    integer vars;\n" after
    func_pattern = re.compile(
        r'(function\s+automatic\b[^;]*;\n)', re.MULTILINE
    )
    offset = 0
    for m in func_pattern.finditer(content):
        # Find which var names are used in this function
        # Get function body (up to endfunction)
        func_start = m.end() + offset
        func_end_match = re.search(r'\bendfunction\b', content[func_start:])
        if not func_end_match:
            continue
        func_body = content[func_start:func_start + func_end_match.start()]
        func_vars = set()
        for v in var_names:
            if re.search(rf'\bfor\s*\(\s*{v}\s*=', func_body):
                func_vars.add(v)
        if func_vars:
            insert = '    integer ' + ', '.join(sorted(func_vars)) + ';\n'
            content = content[:func_start] + insert + content[func_start:]
            offset += len(insert)

    return content


def fix_file(filepath):
    """Apply all iverilog 10.1 compatibility fixes."""
    with open(filepath, 'r') as f:
        content = f.read()
    original = content

    # === PHASE 1: Type casts (BEFORE type name replacement) ===

    # acc_t'(expr) -> to_acc(expr)
    content = replace_type_cast(content, 'acc_t', lambda e: f'to_acc({e})')
    # data_t'(expr) -> to_data(expr)
    content = replace_type_cast(content, 'data_t', lambda e: f'to_data({e})')
    # int'(expr) -> (expr)
    content = replace_type_cast(content, 'int', lambda e: f'({e})')
    # real'(expr) -> $itor(expr)
    content = replace_type_cast(content, 'real', lambda e: f'$itor({e})')
    # 4'(expr) -> (expr)
    content = re.sub(r"\b4'\(", '(', content)
    # ADDR_W'(expr) -> (expr)
    content = re.sub(r"\bADDR_W'\(", '(', content)

    # === PHASE 2: Enum typedefs -> localparams ===
    content = fix_enums(content)

    # === PHASE 3: Replace type names ===

    # In input/output declarations
    # iverilog needs "output reg" for outputs assigned in always blocks
    # Use "output reg" for all typed outputs (safe since they're all registered)
    content = re.sub(r'(input\s+)data_t\b', r'\1signed [15:0]', content)
    content = re.sub(r'(output\s+)data_t\b', r'\1reg signed [15:0]', content)
    content = re.sub(r'(input\s+)acc_t\b', r'\1signed [31:0]', content)
    content = re.sub(r'(output\s+)acc_t\b', r'\1reg signed [31:0]', content)
    content = re.sub(r'(input\s+)seq_idx_t\b', r'\1[6:0]', content)
    content = re.sub(r'(output\s+)seq_idx_t\b', r'\1reg [6:0]', content)

    # In standalone declarations: "data_t name" -> "reg signed [15:0] name"
    # But NOT in function signatures (those have input/output already)
    # Match lines starting with optional spaces + data_t
    content = re.sub(r'^(\s+)data_t\b', r'\1reg signed [15:0]', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)acc_t\b', r'\1reg signed [31:0]', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)seq_idx_t\b', r'\1reg [6:0]', content, flags=re.MULTILINE)

    # In localparam/parameter: "localparam data_t NAME" -> "localparam signed [15:0] NAME"
    content = re.sub(r'(localparam\s+)data_t\b', r'\1signed [15:0]', content)

    # Remaining data_t in other contexts (function return types handled by package rewrite)
    # "data_t name [" -> "reg signed [15:0] name ["  (unpacked array declarations)
    # Already handled by the standalone declaration regex above

    # Fix seq_idx_t that appears inline (not at start of line)
    content = re.sub(r'(?<=\s)seq_idx_t\b', '[6:0]', content)

    # Fix function return types: "function automatic data_t name" etc.
    content = re.sub(r'function\s+automatic\s+data_t\b', 'function automatic signed [15:0]', content)
    content = re.sub(r'function\s+automatic\s+acc_t\b', 'function automatic signed [31:0]', content)

    # === PHASE 4: Syntax fixes ===

    # always_ff -> always
    content = re.sub(r'\balways_ff\s+@', 'always @', content)
    # always_comb -> always @(*)
    content = re.sub(r'\balways_comb\b', 'always @(*)', content)
    # assert -> if
    content = re.sub(r'\bassert\s*\(', 'if (', content)
    # '0 fill pattern -> 0
    content = re.sub(r"(?<![0-9a-zA-Z_'])'0\b", '0', content)
    # task automatic name() -> task automatic name;
    content = re.sub(r'task\s+automatic\s+(\w+)\(\s*\)\s*;', r'task automatic \1;', content)
    # Fix task calls with empty parens: reset() -> reset
    content = re.sub(r'\breset\(\)', 'reset', content)

    # Remove generate/endgenerate keywords (iverilog 10.1 doesn't support them)
    content = re.sub(r'^\s*generate\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*endgenerate\s*$', '', content, flags=re.MULTILINE)

    # Fix genvar ++ increment: gi++ -> gi = gi + 1 (only in for loops with genvar-style vars)
    content = re.sub(r'\b(\w+)\+\+\)', r'\1 = \1 + 1)', content)

    # output logic that gets assigned in always blocks needs to be output reg
    # In iverilog, "output logic" is treated as wire. Convert to "output reg"
    content = content.replace('output logic  ', 'output reg  ')
    content = content.replace('output logic ', 'output reg ')

    # === PHASE 4b: Remove parameter part-selects ===
    # iverilog 10.1 crashes on PARAM[$clog2(PARAM):0] and PARAM[N:0] syntax.
    # These are just width-truncation casts and can be safely removed.
    # Only match when NOT preceded by '[' (to avoid matching inside width specs)
    # Pattern: UPPERCASE_NAME[$clog2(...)...] -> UPPERCASE_NAME
    content = re.sub(r'(?<!\[)\b([A-Z_]{2,})\[\$clog2\([^)]+\)\s*:\s*0\]', r'\1', content)
    content = re.sub(r'(?<!\[)\b([A-Z_]{2,})\[\$clog2\([^)]+\)\s*-\s*1\s*:\s*0\]', r'\1', content)
    # PARAM[N:0] pattern like D_FF[8:0] — only when used as value (not in declarations)
    # Match when preceded by word boundary but NOT by '['
    content = re.sub(r'(?<!\[)\b([A-Z_]{2,})\[(\d+):0\]', r'\1', content)

    # === PHASE 4c: Move inline reg/int declarations out of always blocks ===
    # iverilog 10.1 doesn't support variable declarations inside always/case blocks.
    # Move them to module level and replace inline initializations with assignments.

    # Find inline "reg signed [N:0] name = value;" -> module-level decl + assignment
    # Pattern: indented reg inside always block (more than 2 levels of indent)
    inline_decls = []

    # Match: deeply indented "reg signed [31:0] name = value;" (6+ spaces = inside always)
    # or:    "reg signed [31:0] name, name2, name3;"
    for m in re.finditer(
        r'^( {6,})reg\s+(signed\s+\[\d+:\d+\])\s+(\w+(?:\s*,\s*\w+)*)\s*(?:=\s*(\S+))?\s*;',
        content, re.MULTILINE
    ):
        indent = m.group(1)
        type_spec = m.group(2)
        names = m.group(3)
        init_val = m.group(4)
        name_list = [n.strip() for n in names.split(',')]
        for name in name_list:
            inline_decls.append(('reg ' + type_spec, name))
        if init_val:
            # Replace entire line with just assignments
            assigns = '; '.join(f'{n} = {init_val}' for n in name_list) + ';'
            content = content.replace(m.group(0), indent + assigns)
        else:
            # Remove the declaration line entirely (just whitespace remains)
            content = content.replace(m.group(0), '')

    # Match: "                int name = expr;" inside procedural blocks (6+ spaces)
    for m in re.finditer(
        r'^( {6,})int\s+(\w+)\s*=\s*([^;]+);',
        content, re.MULTILINE
    ):
        indent = m.group(1)
        name = m.group(2)
        expr = m.group(3).strip()
        inline_decls.append(('integer', name))
        content = content.replace(m.group(0), f'{indent}{name} = {expr};')

    # Insert collected declarations at module level (after port list)
    if inline_decls:
        # Deduplicate
        seen = set()
        unique_decls = []
        for dtype, name in inline_decls:
            if name not in seen:
                seen.add(name)
                unique_decls.append((dtype, name))
        port_close = re.search(r'\);\s*\n', content)
        if port_close:
            decl_lines = '\n'.join(f'  {dtype} {name};' for dtype, name in unique_decls) + '\n'
            insert_pos = port_close.end()
            content = content[:insert_pos] + decl_lines + content[insert_pos:]

    # === PHASE 5: Inline to_data/to_acc in module-local functions ===
    # iverilog 10.1 can't resolve package functions called from module-local functions.
    # Replace to_data(expr) with (expr)[15:0] and to_acc(expr) with sign-extended form
    # ONLY inside module-local function bodies (between function...endfunction within module)
    if re.search(r'^\s+function\s+automatic', content, re.MULTILINE):
        # This module has local functions - inline to_data/to_acc calls within them
        # Simple approach: replace to_data(X) with ((X)[15:0]) and to_acc(X) with
        # {{16{(X)[15]}}, (X)} throughout the file. These are equivalent.
        # Actually safer: just add local copies of to_data/to_acc inside modules that
        # have local functions calling them.
        # Simplest: check if module has functions AND uses to_data/to_acc
        has_local_funcs = bool(re.search(
            r'^\s+function\s+automatic\s+signed\s+\[',
            content, re.MULTILINE))
        uses_helpers = 'to_data(' in content or 'to_acc(' in content
        if has_local_funcs and uses_helpers:
            # Check if module already has to_data defined (from package import it won't)
            if 'function automatic signed [15:0] to_data' not in content:
                # Insert local copies after port list
                port_close = re.search(r'\);\s*\n', content)
                if port_close:
                    helpers = (
                        '\n'
                        '  // Local copies of to_data/to_acc (iverilog 10.1 can\'t resolve\n'
                        '  // package functions from within module-local functions)\n'
                        '  function automatic signed [31:0] to_acc(input signed [15:0] val);\n'
                        '    to_acc = {{16{val[15]}}, val};\n'
                        '  endfunction\n'
                        '  function automatic signed [15:0] to_data(input signed [31:0] val);\n'
                        '    to_data = val[15:0];\n'
                        '  endfunction\n'
                    )
                    insert_pos = port_close.end()
                    content = content[:insert_pos] + helpers + content[insert_pos:]

    # === PHASE 6: Replace for(int ...) with integer declarations ===
    # iverilog 10.1 segfaults on `int` loop vars used in multiply-index expressions
    content = fix_for_int_loops(content)

    # === PHASE 6: Cleanup ===
    # Fix double "logic logic" or "reg logic" etc.
    content = re.sub(r'\blogic\s+logic\b', 'logic', content)
    content = re.sub(r'\breg\s+logic\b', 'reg', content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def rewrite_package(filepath):
    """Rewrite transformer_pkg.sv for iverilog 10.1 compatibility."""
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. parameter -> localparam
    content = content.replace('  parameter int ', '  localparam int ')
    content = content.replace('  parameter     ', '  localparam     ')

    # 2. Remove typedef lines
    typedef_lines = [
        r'  typedef logic signed \[DATA_WIDTH-1:0\]\s+data_t;[^\n]*\n',
        r'  typedef logic signed \[ACC_WIDTH-1:0\]\s+acc_t;[^\n]*\n',
        r'  typedef logic \[\$clog2\(MAX_SEQ_LEN\)-1:0\]\s+seq_idx_t;[^\n]*\n',
        r'  typedef logic \[\$clog2\(VOCAB_SIZE\)-1:0\]\s+vocab_idx_t;[^\n]*\n',
        r'  typedef logic signed \[D_MODEL-1:0\].*\n',
        r'  typedef logic signed \[D_FF-1:0\].*\n',
        r'  typedef logic signed \[MAX_SEQ_LEN-1:0\].*\n',
    ]
    for pat in typedef_lines:
        content = re.sub(pat, '', content)

    # 3. Fix function signatures
    content = content.replace('function automatic data_t ', 'function automatic signed [15:0] ')
    content = content.replace('function automatic logic [3:0] ', 'function automatic [3:0] ')
    content = content.replace('function automatic logic [15:0] ', 'function automatic [15:0] ')
    content = content.replace('(input data_t ', '(input signed [15:0] ')
    content = content.replace(', input data_t ', ', input signed [15:0] ')
    content = content.replace('(input int val)', '(input signed [31:0] val)')

    # 4. Fix data_t' casts inside the package
    content = replace_type_cast(content, 'data_t', lambda e: f'to_data({e})')

    # 5. Remaining data_t variable declarations inside functions
    content = re.sub(r'^(\s+)data_t\b', r'\1reg signed [15:0]', content, flags=re.MULTILINE)

    # 6. Add helper functions
    if 'function automatic signed [31:0] to_acc' not in content:
        marker = '  // =========================================================================\n  // Fixed-Point Utility Functions'
        helpers = """  // =========================================================================
  // Type Cast Helpers (iverilog 10.1 compatibility)
  // =========================================================================
  function automatic signed [31:0] to_acc(input signed [15:0] val);
    to_acc = {{16{val[15]}}, val};
  endfunction

  function automatic signed [15:0] to_data(input signed [31:0] val);
    to_data = val[15:0];
  endfunction

"""
        content = content.replace(marker, helpers + marker)

    with open(filepath, 'w') as f:
        f.write(content)


def main():
    rtl_dir = os.path.join(PROJ_ROOT, 'rtl')
    tb_dir = os.path.join(PROJ_ROOT, 'tb', 'sv')

    # Step 1: Rewrite the package first
    pkg_path = os.path.join(rtl_dir, 'transformer_pkg.sv')
    rewrite_package(pkg_path)
    print("  Rewrote transformer_pkg.sv")

    # Step 2: Fix all files (including the package for remaining fixes)
    files = []
    for d in [rtl_dir, tb_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.sv'):
                    files.append(os.path.join(d, f))

    for filepath in sorted(files):
        modified = fix_file(filepath)
        status = "MODIFIED" if modified else "unchanged"
        print(f"  {status}: {os.path.basename(filepath)}")


if __name__ == '__main__':
    main()
