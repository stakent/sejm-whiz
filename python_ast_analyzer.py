#!/usr/bin/env python3
"""
Comprehensive Python AST Analyzer for Dead Code Detection
Analyzes the entire Python codebase to identify function definitions, class definitions,
imports, and usage patterns to facilitate dead code detection.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import argparse


@dataclass
class FunctionDef:
    """Represents a function or method definition"""

    name: str
    file_path: str
    line_number: int
    is_method: bool
    class_name: Optional[str]
    args: List[str]
    decorators: List[str]
    is_async: bool
    docstring: Optional[str]
    is_private: bool
    is_dunder: bool


@dataclass
class ClassDef:
    """Represents a class definition"""

    name: str
    file_path: str
    line_number: int
    base_classes: List[str]
    decorators: List[str]
    methods: List[str]
    docstring: Optional[str]
    is_private: bool


@dataclass
class ImportDef:
    """Represents an import statement"""

    module: str
    name: Optional[str]  # None for 'import module', name for 'from module import name'
    alias: Optional[str]
    file_path: str
    line_number: int
    is_relative: bool


@dataclass
class Usage:
    """Represents usage of a function, class, or variable"""

    name: str
    file_path: str
    line_number: int
    context: str  # 'call', 'attribute', 'name'


@dataclass
class AnalysisResult:
    """Complete analysis result"""

    functions: List[FunctionDef]
    classes: List[ClassDef]
    imports: List[ImportDef]
    usages: List[Usage]
    file_stats: Dict[str, Any]
    potential_dead_code: Dict[str, List[str]]


class PythonASTAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code structure and usage patterns"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions: List[FunctionDef] = []
        self.classes: List[ClassDef] = []
        self.imports: List[ImportDef] = []
        self.usages: List[Usage] = []
        self.current_class = None
        self.defined_names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition"""
        is_method = self.current_class is not None
        class_name = self.current_class.name if self.current_class else None

        # Extract function arguments
        args = []
        if node.args.args:
            args = [arg.arg for arg in node.args.args]

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
            else:
                decorators.append(ast.unparse(decorator))

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Determine if private or dunder method
        is_private = node.name.startswith("_") and not node.name.startswith("__")
        is_dunder = node.name.startswith("__") and node.name.endswith("__")

        func_def = FunctionDef(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            is_method=is_method,
            class_name=class_name,
            args=args,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=docstring,
            is_private=is_private,
            is_dunder=is_dunder,
        )

        self.functions.append(func_def)
        self.defined_names.add(node.name)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition"""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition"""
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            else:
                base_classes.append(ast.unparse(base))

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            else:
                decorators.append(ast.unparse(decorator))

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Determine if private
        is_private = node.name.startswith("_")

        # Store current class context
        old_class = self.current_class
        self.current_class = node

        # Extract method names (will be populated as we visit methods)
        methods = []

        class_def = ClassDef(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            decorators=decorators,
            methods=methods,  # Will be populated later
            docstring=docstring,
            is_private=is_private,
        )

        self.classes.append(class_def)
        self.defined_names.add(node.name)

        # Visit class body
        self.generic_visit(node)

        # Extract method names from functions defined in this class
        class_methods = [
            f.name
            for f in self.functions
            if f.class_name == node.name and f.file_path == self.file_path
        ]
        class_def.methods = class_methods

        # Restore previous class context
        self.current_class = old_class

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement"""
        for alias in node.names:
            import_def = ImportDef(
                module=alias.name,
                name=None,
                alias=alias.asname,
                file_path=self.file_path,
                line_number=node.lineno,
                is_relative=False,
            )
            self.imports.append(import_def)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from ... import ... statement"""
        if node.module is None:
            return

        for alias in node.names:
            import_def = ImportDef(
                module=node.module,
                name=alias.name,
                alias=alias.asname,
                file_path=self.file_path,
                line_number=node.lineno,
                is_relative=node.level > 0,
            )
            self.imports.append(import_def)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call"""
        if isinstance(node.func, ast.Name):
            usage = Usage(
                name=node.func.id,
                file_path=self.file_path,
                line_number=node.lineno,
                context="call",
            )
            self.usages.append(usage)
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like obj.method()
            if isinstance(node.func.value, ast.Name):
                usage = Usage(
                    name=f"{node.func.value.id}.{node.func.attr}",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    context="call",
                )
                self.usages.append(usage)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name reference"""
        if isinstance(node.ctx, ast.Load):
            usage = Usage(
                name=node.id,
                file_path=self.file_path,
                line_number=node.lineno,
                context="name",
            )
            self.usages.append(usage)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access"""
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            usage = Usage(
                name=f"{node.value.id}.{node.attr}",
                file_path=self.file_path,
                line_number=node.lineno,
                context="attribute",
            )
            self.usages.append(usage)

        self.generic_visit(node)


class CodebaseAnalyzer:
    """Main analyzer for the entire codebase"""

    def __init__(self, root_path: str, exclude_patterns: List[str] = None):
        self.root_path = Path(root_path)
        self.exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".pyc",
            "test/",
            "tests/",
            ".git",
            ".venv",
            "venv/",
            "env/",
            ".tox",
            "build/",
            "dist/",
        ]
        self.logger = logging.getLogger(__name__)

    def should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis"""
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True
        return False

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase"""
        python_files = []
        for file_path in self.root_path.rglob("*.py"):
            if not self.should_exclude_file(file_path):
                python_files.append(file_path)
        return sorted(python_files)

    def analyze_file(self, file_path: Path) -> Optional[PythonASTAnalyzer]:
        """Analyze a single Python file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            analyzer = PythonASTAnalyzer(str(file_path))
            analyzer.visit(tree)
            return analyzer

        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.warning(f"Could not parse {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def analyze_codebase(self) -> AnalysisResult:
        """Analyze the entire codebase"""
        self.logger.info(f"Starting codebase analysis of {self.root_path}")

        python_files = self.find_python_files()
        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        all_functions = []
        all_classes = []
        all_imports = []
        all_usages = []
        file_stats = {}

        for file_path in python_files:
            self.logger.debug(f"Analyzing {file_path}")
            analyzer = self.analyze_file(file_path)

            if analyzer:
                all_functions.extend(analyzer.functions)
                all_classes.extend(analyzer.classes)
                all_imports.extend(analyzer.imports)
                all_usages.extend(analyzer.usages)

                file_stats[str(file_path)] = {
                    "functions": len(analyzer.functions),
                    "classes": len(analyzer.classes),
                    "imports": len(analyzer.imports),
                    "usages": len(analyzer.usages),
                }

        # Analyze potential dead code
        potential_dead_code = self.identify_potential_dead_code(
            all_functions, all_classes, all_usages
        )

        result = AnalysisResult(
            functions=all_functions,
            classes=all_classes,
            imports=all_imports,
            usages=all_usages,
            file_stats=file_stats,
            potential_dead_code=potential_dead_code,
        )

        self.logger.info(
            f"Analysis complete: {len(all_functions)} functions, "
            f"{len(all_classes)} classes, {len(all_imports)} imports, "
            f"{len(all_usages)} usages analyzed"
        )

        return result

    def identify_potential_dead_code(
        self, functions: List[FunctionDef], classes: List[ClassDef], usages: List[Usage]
    ) -> Dict[str, List[str]]:
        """Identify potentially unused functions and classes"""

        # Extract used names from various contexts
        used_names = set()
        for usage in usages:
            # Handle simple names
            used_names.add(usage.name)
            # Handle dotted names (extract the first part)
            if "." in usage.name:
                used_names.add(usage.name.split(".")[0])

        # Find potentially unused functions
        unused_functions = []
        for func in functions:
            # Skip certain types of functions that are commonly used but not directly called
            if (
                func.is_dunder  # Double underscore methods
                or func.name in ["main", "setUp", "tearDown"]  # Special methods
                or func.decorators  # Decorated functions (could be used by framework)
                or func.name.startswith("test_")
            ):  # Test functions
                continue

            if func.name not in used_names:
                unused_functions.append(
                    f"{func.file_path}:{func.line_number}:{func.name}"
                )

        # Find potentially unused classes
        unused_classes = []
        for cls in classes:
            # Skip exception classes and other special classes
            if (
                cls.name.endswith("Error")
                or cls.name.endswith("Exception")
                or cls.decorators
                or "Abstract" in cls.name
            ):
                continue

            if cls.name not in used_names:
                unused_classes.append(f"{cls.file_path}:{cls.line_number}:{cls.name}")

        return {"unused_functions": unused_functions, "unused_classes": unused_classes}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Python AST Analyzer for Dead Code Detection"
    )
    parser.add_argument("path", help="Root path to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument(
        "--include-tests", action="store_true", help="Include test files in analysis"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    exclude_patterns = [
        "__pycache__",
        ".pyc",
        ".git",
        ".venv",
        "venv/",
        "env/",
        ".tox",
        "build/",
        "dist/",
    ]

    if not args.include_tests:
        exclude_patterns.extend(["test/", "tests/"])

    analyzer = CodebaseAnalyzer(args.path, exclude_patterns)
    result = analyzer.analyze_codebase()

    # Convert result to JSON-serializable format
    result_dict = {
        "functions": [asdict(f) for f in result.functions],
        "classes": [asdict(c) for c in result.classes],
        "imports": [asdict(i) for i in result.imports],
        "usages": [asdict(u) for u in result.usages],
        "file_stats": result.file_stats,
        "potential_dead_code": result.potential_dead_code,
        "summary": {
            "total_functions": len(result.functions),
            "total_classes": len(result.classes),
            "total_imports": len(result.imports),
            "total_usages": len(result.usages),
            "files_analyzed": len(result.file_stats),
            "potential_unused_functions": len(
                result.potential_dead_code["unused_functions"]
            ),
            "potential_unused_classes": len(
                result.potential_dead_code["unused_classes"]
            ),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Analysis results written to {args.output}")
    else:
        # Print summary to stdout
        print("\n=== Python AST Analysis Results ===")
        print(f"Files analyzed: {result_dict['summary']['files_analyzed']}")
        print(f"Functions found: {result_dict['summary']['total_functions']}")
        print(f"Classes found: {result_dict['summary']['total_classes']}")
        print(f"Imports found: {result_dict['summary']['total_imports']}")
        print(f"Usages found: {result_dict['summary']['total_usages']}")
        print(
            f"Potentially unused functions: {result_dict['summary']['potential_unused_functions']}"
        )
        print(
            f"Potentially unused classes: {result_dict['summary']['potential_unused_classes']}"
        )

        if result.potential_dead_code["unused_functions"]:
            print("\n=== Potentially Unused Functions ===")
            for func in result.potential_dead_code["unused_functions"][
                :20
            ]:  # Limit to first 20
                print(f"  {func}")
            if len(result.potential_dead_code["unused_functions"]) > 20:
                print(
                    f"  ... and {len(result.potential_dead_code['unused_functions']) - 20} more"
                )

        if result.potential_dead_code["unused_classes"]:
            print("\n=== Potentially Unused Classes ===")
            for cls in result.potential_dead_code["unused_classes"][
                :20
            ]:  # Limit to first 20
                print(f"  {cls}")
            if len(result.potential_dead_code["unused_classes"]) > 20:
                print(
                    f"  ... and {len(result.potential_dead_code['unused_classes']) - 20} more"
                )


if __name__ == "__main__":
    main()
