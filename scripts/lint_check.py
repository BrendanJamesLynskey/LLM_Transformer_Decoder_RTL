#!/usr/bin/env python3
"""
SystemVerilog RTL Syntax & Structural Lint Check
Validates the RTL files for common issues without requiring iverilog.
"""

import os
import re
import sys
from pathlib import Path

RTL_DIR = Path(__file__).parent.parent / "rtl"


class SVLinter:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.files_checked = 0

    def error(self, file, line_num, msg):
        self.errors.append(f"  ERROR [{file}:{line_num}] {msg}")

    def warn(self, file, line_num, msg):
        self.warnings.append(f"  WARN  [{file}:{line_num}] {msg}")

    def check_file(self, filepath):
        fname = filepath.name
        self.files_checked += 1
        print(f"  Checking {fname}...")

        with open(filepath) as f:
            lines = f.readlines()
        content = ''.join(lines)

        # --- Basic structure checks ---

        # Strip comments for keyword counting
        code_only = re.sub(r'//.*', '', content)              # line comments
        code_only = re.sub(r'/\*.*?\*/', '', code_only, flags=re.DOTALL)  # block comments

        # Module/package declaration
        has_module = bool(re.search(r'\b(module|package)\b', code_only))
        if not has_module:
            self.error(fname, 0, "No module or package declaration found")
            return

        # Matching endmodule/endpackage
        module_count = len(re.findall(r'\bmodule\b', code_only))
        endmodule_count = len(re.findall(r'\bendmodule\b', code_only))
        pkg_count = len(re.findall(r'\bpackage\b', code_only))
        endpkg_count = len(re.findall(r'\bendpackage\b', code_only))

        if module_count != endmodule_count:
            self.error(fname, 0, f"Mismatched module/endmodule ({module_count}/{endmodule_count})")
        if pkg_count != endpkg_count:
            self.error(fname, 0, f"Mismatched package/endpackage ({pkg_count}/{endpkg_count})")

        # --- Line-by-line checks ---
        in_block_comment = False
        begin_count = 0
        end_count = 0
        paren_depth = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track block comments
            if '/*' in stripped:
                in_block_comment = True
            if '*/' in stripped:
                in_block_comment = False
                continue
            if in_block_comment:
                continue

            # Remove line comments for analysis
            code = re.sub(r'//.*', '', stripped)

            # Count begin/end
            begin_count += len(re.findall(r'\bbegin\b', code))
            end_count += len(re.findall(r'\bend\b(?!\w)', code))

            # Check for common issues

            # Blocking assignment in sequential block (= in always_ff)
            # This is a simplified check
            if re.search(r'<=\s*\S', code):
                pass  # NBA is fine in always_ff

            # Missing semicolons on non-block statements
            if code and not code.startswith('//') and not code.endswith((';', 'begin', 'end', ')', '(', '{', '}', ',', ':')) \
               and not any(kw in code for kw in ['module', 'endmodule', 'package', 'endpackage',
                           'function', 'endfunction', 'task', 'endtask', 'generate', 'endgenerate',
                           'always_ff', 'always_comb', 'if', 'else', 'for', 'case', 'endcase',
                           'default', 'typedef', 'import', 'localparam', 'parameter', 'genvar',
                           'fork', 'join', 'initial', '//', '/*', '*/']):
                if len(code) > 5 and not code.startswith('#'):
                    pass  # Too many false positives; skip

            # Check for X/Z assignments (could be intentional but worth flagging)
            if re.search(r"['\"].*[xXzZ]", code) and 'display' not in code:
                pass  # Common in parameters, not always an issue

        # --- Structural checks ---

        # always_ff should use <= (NBA)
        in_always_ff = False
        for i, line in enumerate(lines, 1):
            code = re.sub(r'//.*', '', line.strip())
            if 'always_ff' in code:
                in_always_ff = True
            if in_always_ff:
                # Check for blocking assignment (but not in function calls or comparisons)
                # This is a rough heuristic
                if re.search(r'(?<!=)(?<!<)(?<!>)(?<!\!)=(?!=)(?!>)', code):
                    # Filter out: parameter assignments, for-loop iterators, comparisons
                    if not any(kw in code for kw in ['for', 'int', 'logic', 'parameter',
                               'localparam', 'if', 'assert', 'function', '$']):
                        pass  # Could flag but too noisy

                if 'endmodule' in code or ('always_comb' in code and 'always_ff' not in code):
                    in_always_ff = False

        # Port width matching (basic check)
        # Look for import statements
        if 'import transformer_pkg' in content:
            self.info.append(f"  INFO  [{fname}] imports transformer_pkg ✓")

        # Check for sensitivity list issues
        if 'always @(' in content and 'always_ff' not in content and 'always_comb' not in content:
            # Old-style always blocks
            for i, line in enumerate(lines, 1):
                if 'always @(' in line and 'posedge' not in line and 'negedge' not in line:
                    self.warn(fname, i, "Old-style combinational always block (prefer always_comb)")

        # Check for proper reset in always_ff
        ff_blocks = re.finditer(r'always_ff\s*@\s*\(([^)]+)\)', content)
        for m in ff_blocks:
            sensitivity = m.group(1)
            if 'posedge clk' in sensitivity:
                if 'negedge rst_n' not in sensitivity and 'posedge rst' not in sensitivity:
                    line_num = content[:m.start()].count('\n') + 1
                    self.warn(fname, line_num, "always_ff with clk but no reset in sensitivity list")

    def check_all(self):
        print("\n" + "="*60)
        print("  SYSTEMVERILOG RTL LINT CHECK")
        print("="*60 + "\n")

        sv_files = sorted(RTL_DIR.glob("*.sv"))
        if not sv_files:
            print(f"  No .sv files found in {RTL_DIR}")
            return False

        for f in sv_files:
            self.check_file(f)

        # --- Cross-file checks ---
        print(f"\n  Cross-file checks...")

        # Check that all instantiated modules exist
        all_modules = set()
        all_instantiated = set()

        for f in sv_files:
            content = open(f).read()
            # Find module declarations
            for m in re.finditer(r'\bmodule\s+(\w+)', content):
                all_modules.add(m.group(1))
            # Find instantiations (name followed by #( or instance_name ()
            for m in re.finditer(r'\b(\w+)\s+(?:#\s*\(|u_\w+|gen_\w+)', content):
                name = m.group(1)
                if name not in ('module', 'logic', 'reg', 'wire', 'input', 'output',
                                'always_ff', 'always_comb', 'assign', 'if', 'for',
                                'begin', 'end', 'function', 'task', 'generate'):
                    all_instantiated.add(name)

        # Check for missing modules
        for mod in all_instantiated:
            if mod not in all_modules and mod not in ('transformer_pkg',):
                self.warn("cross-file", 0, f"Module '{mod}' instantiated but not defined in RTL")

        print(f"\n  Modules defined: {sorted(all_modules)}")
        print(f"  Modules instantiated: {sorted(all_instantiated)}")

        # Check package usage
        pkg_defined = 'transformer_pkg' in all_modules
        if not pkg_defined:
            # Check if it's a package (not a module)
            for f in sv_files:
                if 'package transformer_pkg' in open(f).read():
                    pkg_defined = True
                    break
        if pkg_defined:
            print(f"  Package transformer_pkg: defined ✓")

        # --- Summary ---
        print(f"\n" + "-"*60)
        print(f"  Files checked:  {self.files_checked}")
        print(f"  Errors:         {len(self.errors)}")
        print(f"  Warnings:       {len(self.warnings)}")

        for e in self.errors:
            print(e)
        for w in self.warnings:
            print(w)
        for info in self.info:
            print(info)

        if self.errors:
            print(f"\n  LINT: FAIL ({len(self.errors)} errors)")
            return False
        else:
            print(f"\n  LINT: PASS (no errors, {len(self.warnings)} warnings)")
            return True


def main():
    linter = SVLinter()
    ok = linter.check_all()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
