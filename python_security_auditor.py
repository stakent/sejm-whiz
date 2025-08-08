#!/usr/bin/env python3
"""
Python Security Auditor Agent

A comprehensive security analysis tool for Python codebases that identifies vulnerabilities,
security anti-patterns, and provides actionable remediation guidance.

Features:
- Static security analysis with bandit integration
- Dependency vulnerability scanning with safety
- Custom security pattern detection
- OWASP Top 10 mapping
- CWE classification
- Advanced taint analysis
- Security best practices enforcement
- Comprehensive reporting with multiple output formats

Author: Claude Code Security Analysis Framework
"""

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class Severity(Enum):
    """Security finding severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""

    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    DESERIALIZATION = "DESERIALIZATION"
    AUTH_BYPASS = "AUTH_BYPASS"
    CRYPTO_WEAKNESS = "CRYPTO_WEAKNESS"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    DANGEROUS_FUNCTION = "DANGEROUS_FUNCTION"
    INSECURE_RANDOM = "INSECURE_RANDOM"
    HARDCODED_SECRET = "HARDCODED_SECRET"
    FILE_OPERATION = "FILE_OPERATION"
    NETWORK_SECURITY = "NETWORK_SECURITY"
    INFO_DISCLOSURE = "INFO_DISCLOSURE"
    DEPENDENCY_VULN = "DEPENDENCY_VULN"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    RACE_CONDITION = "RACE_CONDITION"
    TOCTOU = "TOCTOU"
    MEMORY_SAFETY = "MEMORY_SAFETY"


@dataclass
class SecurityFinding:
    """Represents a security vulnerability or issue."""

    id: str
    severity: Severity
    vulnerability_type: VulnerabilityType
    title: str
    description: str
    file_path: str
    line_number: int
    column: int = 0
    code_snippet: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    cvss_score: Optional[float] = None
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    test_id: Optional[str] = None
    false_positive: bool = False


@dataclass
class DependencyVulnerability:
    """Represents a vulnerability in a dependency."""

    package: str
    version: str
    vulnerability_id: str
    advisory: str
    affected_versions: str
    fixed_version: Optional[str] = None
    severity: Severity = Severity.MEDIUM


@dataclass
class SecurityReport:
    """Comprehensive security analysis report."""

    scan_timestamp: datetime
    target_path: str
    findings: List[SecurityFinding]
    dependency_vulnerabilities: List[DependencyVulnerability]
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    false_positives: int = 0
    scan_duration: float = 0.0


class SecurityPatterns:
    """Security pattern definitions and detection rules."""

    # Dangerous function patterns
    DANGEROUS_FUNCTIONS = {
        "eval": {
            "severity": Severity.CRITICAL,
            "message": "Use of eval() can lead to arbitrary code execution",
            "remediation": "Replace eval() with ast.literal_eval() for safe evaluation of literals, or use a safer alternative",
        },
        "exec": {
            "severity": Severity.CRITICAL,
            "message": "Use of exec() can lead to arbitrary code execution",
            "remediation": "Avoid exec() or implement strict input validation and sandboxing",
        },
        "compile": {
            "severity": Severity.HIGH,
            "message": "Dynamic code compilation can be dangerous",
            "remediation": "Validate and sanitize code before compilation, consider alternatives",
        },
        "__import__": {
            "severity": Severity.HIGH,
            "message": "Dynamic imports can lead to module injection",
            "remediation": "Use importlib with validation or whitelist allowed modules",
        },
    }

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'execute\s*\(\s*["\'].*%.*["\']',  # String formatting in SQL
        r"execute\s*\(\s*.*\+.*\)",  # String concatenation in SQL
        r'executemany\s*\(\s*["\'].*%.*["\']',
        r'raw\s*\(\s*["\'].*%.*["\']',  # Django raw queries
        r"extra\s*\(\s*where.*%",  # Django extra with formatting
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"os\.system\s*\(\s*.*\+",  # os.system with concatenation
        r"subprocess\.(call|run|Popen)\s*\(\s*.*\+",  # subprocess with concatenation
        r'subprocess\.(call|run|Popen)\s*\(\s*["\'].*%.*["\']',  # subprocess with formatting
        r"os\.popen\s*\(\s*.*\+",  # os.popen with concatenation
    ]

    # XSS patterns (for web frameworks)
    XSS_PATTERNS = [
        r"render_template_string\s*\(\s*.*\+",  # Flask template injection
        r"Markup\s*\(\s*.*\+",  # Flask Markup with concatenation
        r"format_html\s*\(\s*.*\+",  # Django format_html with concatenation
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"open\s*\(\s*.*\+",  # File open with concatenation
        r"os\.path\.join\s*\(\s*.*user",  # Path join with user input
        r"pathlib\.Path\s*\(\s*.*\+",  # Path construction with concatenation
    ]

    # Hardcoded secrets patterns
    SECRET_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded API key"),
        (r'secret_key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret key"),
        (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token"),
        (r'private_key\s*=\s*["\'][^"\']{100,}["\']', "Hardcoded private key"),
    ]

    # Cryptographic weakness patterns
    CRYPTO_WEAKNESS_PATTERNS = [
        (r"hashlib\.md5\s*\(", "MD5 is cryptographically broken"),
        (r"hashlib\.sha1\s*\(", "SHA1 is cryptographically weak"),
        (r"DES\s*\(", "DES encryption is weak"),
        (r"RC4\s*\(", "RC4 encryption is broken"),
        (
            r"random\.random\s*\(",
            "Use cryptographically secure random for security purposes",
        ),
    ]


class PythonSecurityAuditor:
    """Main security auditor class."""

    def __init__(self, target_path: str, config: Optional[Dict] = None):
        self.target_path = Path(target_path)
        self.config = config or {}
        self.findings: List[SecurityFinding] = []
        self.dependency_vulnerabilities: List[DependencyVulnerability] = []
        self.patterns = SecurityPatterns()

        # Configuration options
        self.include_low_severity = self.config.get("include_low_severity", True)
        self.exclude_test_files = self.config.get("exclude_test_files", True)
        self.custom_patterns = self.config.get("custom_patterns", {})
        self.whitelist_files = self.config.get("whitelist_files", [])
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB

    def audit(self) -> SecurityReport:
        """Perform comprehensive security audit."""
        start_time = datetime.now()

        print("🔍 Starting comprehensive Python security audit...")
        print(f"📁 Target: {self.target_path}")

        # Phase 1: Static analysis with bandit
        print("\n📊 Phase 1: Static security analysis (bandit)")
        self._run_bandit_analysis()

        # Phase 2: Dependency vulnerability scanning
        print("📦 Phase 2: Dependency vulnerability scanning")
        self._scan_dependencies()

        # Phase 3: Custom pattern analysis
        print("🔍 Phase 3: Custom security pattern analysis")
        self._analyze_custom_patterns()

        # Phase 4: Advanced analysis
        print("🧠 Phase 4: Advanced security analysis")
        self._perform_advanced_analysis()

        # Phase 5: Context analysis and false positive reduction
        print("🎯 Phase 5: Context analysis and false positive reduction")
        self._reduce_false_positives()

        # Generate comprehensive report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, duration)

        print(f"\n✅ Security audit completed in {duration:.2f} seconds")
        print(f"🚨 Found {len(self.findings)} security issues")
        print(
            f"📦 Found {len(self.dependency_vulnerabilities)} dependency vulnerabilities"
        )

        return report

    def _run_bandit_analysis(self) -> None:
        """Run bandit security analysis."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                str(self.target_path),
                "-f",
                "json",
                "--skip",
                "B101",  # Skip assert_used test by default
            ]

            if self.exclude_test_files:
                cmd.extend(["--exclude", "**/test_*.py,**/tests/**"])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_data = json.loads(result.stdout)
                    self._process_bandit_results(bandit_data)
                except json.JSONDecodeError:
                    print(f"⚠️  Could not parse bandit output: {result.stdout}")
            else:
                print(f"⚠️  Bandit analysis failed: {result.stderr}")

        except FileNotFoundError:
            print("⚠️  Bandit not found. Install with: pip install bandit")
        except Exception as e:
            print(f"⚠️  Bandit analysis error: {e}")

    def _process_bandit_results(self, bandit_data: Dict) -> None:
        """Process bandit analysis results."""
        for result in bandit_data.get("results", []):
            severity_map = {
                "HIGH": Severity.HIGH,
                "MEDIUM": Severity.MEDIUM,
                "LOW": Severity.LOW,
            }

            severity = severity_map.get(
                result.get("issue_severity", "MEDIUM"), Severity.MEDIUM
            )
            confidence = result.get("issue_confidence", "MEDIUM")

            # Skip low severity if configured
            if not self.include_low_severity and severity == Severity.LOW:
                continue

            finding = SecurityFinding(
                id=f"BANDIT_{result.get('test_id', 'UNKNOWN')}_{hash(result.get('filename', '') + str(result.get('line_number', 0)))}",
                severity=severity,
                vulnerability_type=self._map_bandit_test_to_vuln_type(
                    result.get("test_id", "")
                ),
                title=result.get("test_name", "Security Issue"),
                description=result.get("issue_text", "No description available"),
                file_path=result.get("filename", ""),
                line_number=result.get("line_number", 0),
                code_snippet=result.get("code", ""),
                confidence=confidence,
                test_id=result.get("test_id"),
            )

            # Add CWE and OWASP mapping
            self._enrich_finding_metadata(finding)

            self.findings.append(finding)

    def _scan_dependencies(self) -> None:
        """Scan dependencies for known vulnerabilities."""
        try:
            # Look for requirements files
            req_files = list(self.target_path.glob("**/requirements*.txt"))
            req_files.extend(list(self.target_path.glob("**/pyproject.toml")))

            if not req_files:
                print("📦 No requirements files found, skipping dependency scan")
                return

            # Try safety first
            self._run_safety_scan()

            # Additional dependency analysis
            self._analyze_dependency_files()

        except Exception as e:
            print(f"⚠️  Dependency scanning error: {e}")

    def _run_safety_scan(self) -> None:
        """Run safety vulnerability scan."""
        try:
            cmd = [sys.executable, "-m", "safety", "check", "--json"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.target_path
            )

            if result.returncode == 0:
                print("✅ No known vulnerabilities found in dependencies")
                return

            try:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    dep_vuln = DependencyVulnerability(
                        package=vuln.get("package_name", "unknown"),
                        version=vuln.get("installed_version", "unknown"),
                        vulnerability_id=vuln.get("id", "unknown"),
                        advisory=vuln.get("advisory", "No advisory available"),
                        affected_versions=vuln.get("vulnerable_versions", "unknown"),
                        severity=self._map_safety_severity(vuln.get("id", "")),
                    )
                    self.dependency_vulnerabilities.append(dep_vuln)
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse safety output: {result.stdout}")

        except FileNotFoundError:
            print("⚠️  Safety not found. Install with: pip install safety")
        except Exception as e:
            print(f"⚠️  Safety scan error: {e}")

    def _analyze_dependency_files(self) -> None:
        """Analyze dependency files for security issues."""
        for req_file in self.target_path.glob("**/requirements*.txt"):
            try:
                content = req_file.read_text()

                # Check for unpinned versions
                for line_no, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "==" not in line and ">=" not in line and "~=" not in line:
                            finding = SecurityFinding(
                                id=f"DEP_UNPIN_{hash(str(req_file) + line)}",
                                severity=Severity.LOW,
                                vulnerability_type=VulnerabilityType.SUPPLY_CHAIN,
                                title="Unpinned dependency version",
                                description=f"Dependency '{line}' is not pinned to a specific version",
                                file_path=str(req_file),
                                line_number=line_no,
                                code_snippet=line,
                                remediation="Pin dependencies to specific versions to prevent supply chain attacks",
                            )
                            self.findings.append(finding)

            except Exception as e:
                print(f"⚠️  Error analyzing {req_file}: {e}")

    def _analyze_custom_patterns(self) -> None:
        """Analyze code for custom security patterns."""
        python_files = list(self.target_path.glob("**/*.py"))

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            try:
                if py_file.stat().st_size > self.max_file_size:
                    print(f"⚠️  Skipping large file: {py_file}")
                    continue

                content = py_file.read_text(encoding="utf-8", errors="ignore")
                self._analyze_file_patterns(py_file, content)

            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

    def _analyze_file_patterns(self, file_path: Path, content: str) -> None:
        """Analyze a single file for security patterns."""
        lines = content.splitlines()

        # Analyze each line for patterns
        for line_no, line in enumerate(lines, 1):
            # SQL injection patterns
            for pattern in self.patterns.SQL_INJECTION_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.SQL_INJECTION,
                        "Potential SQL injection vulnerability",
                        "Use parameterized queries or ORM methods",
                        Severity.HIGH,
                    )

            # Command injection patterns
            for pattern in self.patterns.COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.COMMAND_INJECTION,
                        "Potential command injection vulnerability",
                        "Use subprocess with list arguments and avoid shell=True",
                        Severity.HIGH,
                    )

            # XSS patterns
            for pattern in self.patterns.XSS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.XSS,
                        "Potential XSS vulnerability",
                        "Use proper template escaping and avoid string concatenation",
                        Severity.HIGH,
                    )

            # Path traversal patterns
            for pattern in self.patterns.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.PATH_TRAVERSAL,
                        "Potential path traversal vulnerability",
                        "Validate and sanitize file paths, use os.path.abspath()",
                        Severity.MEDIUM,
                    )

            # Hardcoded secrets
            for pattern, message in self.patterns.SECRET_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.HARDCODED_SECRET,
                        f"Hardcoded secret detected: {message}",
                        "Use environment variables or secure configuration management",
                        Severity.HIGH,
                    )

            # Cryptographic weaknesses
            for pattern, message in self.patterns.CRYPTO_WEAKNESS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_pattern_finding(
                        file_path,
                        line_no,
                        line,
                        VulnerabilityType.CRYPTO_WEAKNESS,
                        f"Cryptographic weakness: {message}",
                        "Use stronger cryptographic algorithms (SHA-256, AES)",
                        Severity.MEDIUM,
                    )

        # AST-based analysis for dangerous functions
        try:
            tree = ast.parse(content)
            self._analyze_ast_security(tree, file_path, lines)
        except SyntaxError:
            pass  # Skip files with syntax errors

    def _analyze_ast_security(
        self, tree: ast.AST, file_path: Path, lines: List[str]
    ) -> None:
        """Perform AST-based security analysis."""

        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, auditor):
                self.auditor = auditor

            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.auditor.patterns.DANGEROUS_FUNCTIONS:
                        pattern_info = self.auditor.patterns.DANGEROUS_FUNCTIONS[
                            func_name
                        ]
                        self.auditor._add_pattern_finding(
                            file_path,
                            node.lineno,
                            lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            VulnerabilityType.DANGEROUS_FUNCTION,
                            pattern_info["message"],
                            pattern_info["remediation"],
                            pattern_info["severity"],
                        )

                self.generic_visit(node)

        visitor = SecurityVisitor(self)
        visitor.visit(tree)

    def _perform_advanced_analysis(self) -> None:
        """Perform advanced security analysis."""
        # Taint analysis for tracking user input
        self._perform_taint_analysis()

        # Control flow analysis for security-critical paths
        self._analyze_control_flows()

        # Configuration security analysis
        self._analyze_configuration_security()

    def _perform_taint_analysis(self) -> None:
        """Perform basic taint analysis to track user input."""
        # This is a simplified taint analysis
        # In a production system, this would be much more sophisticated

        taint_sources = [
            "request.args",
            "request.form",
            "request.json",
            "request.data",
            "input()",
            "sys.argv",
            "os.environ",
            "request.GET",
            "request.POST",
        ]

        dangerous_sinks = [
            "execute",
            "eval",
            "exec",
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "open",
        ]

        python_files = list(self.target_path.glob("**/*.py"))

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()

                # Simple pattern-based taint tracking
                for line_no, line in enumerate(lines, 1):
                    # Check if line contains taint source
                    has_taint_source = any(source in line for source in taint_sources)
                    has_dangerous_sink = any(sink in line for sink in dangerous_sinks)

                    if has_taint_source and has_dangerous_sink:
                        self._add_pattern_finding(
                            py_file,
                            line_no,
                            line,
                            VulnerabilityType.INPUT_VALIDATION,
                            "Potential data flow from user input to dangerous function",
                            "Validate and sanitize all user input before using in security-sensitive operations",
                            Severity.HIGH,
                        )

            except Exception as e:
                print(f"⚠️  Error in taint analysis for {py_file}: {e}")

    def _analyze_control_flows(self) -> None:
        """Analyze control flows for security issues."""
        # This would implement control flow graph analysis
        # For now, we'll do basic pattern matching for common issues

        python_files = list(self.target_path.glob("**/*.py"))

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")

                # Check for race conditions in file operations
                if re.search(
                    r"os\.path\.exists.*open\(", content, re.MULTILINE | re.DOTALL
                ):
                    lines = content.splitlines()
                    for line_no, line in enumerate(lines, 1):
                        if "os.path.exists" in line:
                            self._add_pattern_finding(
                                py_file,
                                line_no,
                                line,
                                VulnerabilityType.RACE_CONDITION,
                                "Potential TOCTOU (Time-of-Check-Time-of-Use) vulnerability",
                                "Use try/except blocks instead of checking file existence",
                                Severity.MEDIUM,
                            )
                            break

            except Exception as e:
                print(f"⚠️  Error in control flow analysis for {py_file}: {e}")

    def _analyze_configuration_security(self) -> None:
        """Analyze configuration files for security issues."""
        config_patterns = {
            "DEBUG = True": ("Debug mode enabled in production", Severity.HIGH),
            "ALLOWED_HOSTS = []": ("Empty ALLOWED_HOSTS in Django", Severity.HIGH),
            "SECRET_KEY = ": ("Hardcoded SECRET_KEY", Severity.CRITICAL),
            "ssl_verify = False": ("SSL verification disabled", Severity.HIGH),
            "verify=False": ("Certificate verification disabled", Severity.HIGH),
        }

        config_files = []
        config_files.extend(list(self.target_path.glob("**/settings.py")))
        config_files.extend(list(self.target_path.glob("**/config.py")))
        config_files.extend(list(self.target_path.glob("**/*.conf")))
        config_files.extend(list(self.target_path.glob("**/*.cfg")))

        for config_file in config_files:
            try:
                content = config_file.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()

                for line_no, line in enumerate(lines, 1):
                    for pattern, (message, severity) in config_patterns.items():
                        if pattern in line:
                            self._add_pattern_finding(
                                config_file,
                                line_no,
                                line,
                                VulnerabilityType.INFO_DISCLOSURE,
                                f"Configuration security issue: {message}",
                                "Review configuration for production security",
                                severity,
                            )

            except Exception as e:
                print(f"⚠️  Error analyzing config file {config_file}: {e}")

    def _reduce_false_positives(self) -> None:
        """Reduce false positives using context analysis."""
        for finding in self.findings:
            # Mark test files with lower confidence
            if "test" in finding.file_path.lower():
                if finding.confidence == "HIGH":
                    finding.confidence = "MEDIUM"
                elif finding.confidence == "MEDIUM":
                    finding.confidence = "LOW"

            # Check if finding is in whitelisted files
            for whitelist_pattern in self.whitelist_files:
                if whitelist_pattern in finding.file_path:
                    finding.false_positive = True
                    break

            # Context-specific false positive detection
            self._apply_context_rules(finding)

    def _apply_context_rules(self, finding: SecurityFinding) -> None:
        """Apply context-specific rules to reduce false positives."""
        # Example: SQL injection in migration files might be acceptable
        if finding.vulnerability_type == VulnerabilityType.SQL_INJECTION:
            if "migration" in finding.file_path.lower():
                finding.confidence = "LOW"

        # Example: eval() in configuration parsing might be controlled
        if finding.vulnerability_type == VulnerabilityType.DANGEROUS_FUNCTION:
            if "config" in finding.file_path.lower() and "eval" in finding.code_snippet:
                finding.confidence = "MEDIUM"

    def _generate_report(self, start_time: datetime, duration: float) -> SecurityReport:
        """Generate comprehensive security report."""
        # Calculate summary statistics
        severity_counts = {severity: 0 for severity in Severity}
        vuln_type_counts = {}
        confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        false_positive_count = 0

        for finding in self.findings:
            severity_counts[finding.severity] += 1
            vuln_type_counts[finding.vulnerability_type] = (
                vuln_type_counts.get(finding.vulnerability_type, 0) + 1
            )
            confidence_counts[finding.confidence] += 1
            if finding.false_positive:
                false_positive_count += 1

        # Calculate metrics
        total_files_scanned = len(list(self.target_path.glob("**/*.py")))
        critical_high_issues = (
            severity_counts[Severity.CRITICAL] + severity_counts[Severity.HIGH]
        )

        security_score = max(
            0,
            100
            - (
                severity_counts[Severity.CRITICAL] * 20
                + severity_counts[Severity.HIGH] * 10
                + severity_counts[Severity.MEDIUM] * 5
                + severity_counts[Severity.LOW] * 1
            ),
        )

        summary = {
            "total_findings": len(self.findings),
            "severity_breakdown": {
                sev.value: count for sev, count in severity_counts.items()
            },
            "vulnerability_types": {
                vt.value: count for vt, count in vuln_type_counts.items()
            },
            "confidence_breakdown": confidence_counts,
            "critical_high_issues": critical_high_issues,
            "dependency_vulnerabilities": len(self.dependency_vulnerabilities),
            "security_score": security_score,
            "files_scanned": total_files_scanned,
        }

        metrics = {
            "scan_duration": duration,
            "files_per_second": total_files_scanned / duration if duration > 0 else 0,
            "findings_per_file": len(self.findings) / total_files_scanned
            if total_files_scanned > 0
            else 0,
            "false_positive_rate": false_positive_count / len(self.findings)
            if self.findings
            else 0,
        }

        return SecurityReport(
            scan_timestamp=start_time,
            target_path=str(self.target_path),
            findings=self.findings,
            dependency_vulnerabilities=self.dependency_vulnerabilities,
            summary=summary,
            metrics=metrics,
            false_positives=false_positive_count,
            scan_duration=duration,
        )

    def _add_pattern_finding(
        self,
        file_path: Path,
        line_no: int,
        code_snippet: str,
        vuln_type: VulnerabilityType,
        title: str,
        remediation: str,
        severity: Severity,
    ) -> None:
        """Add a security finding from pattern matching."""
        finding = SecurityFinding(
            id=f"PATTERN_{vuln_type.value}_{hash(str(file_path) + str(line_no) + code_snippet)}",
            severity=severity,
            vulnerability_type=vuln_type,
            title=title,
            description=title,
            file_path=str(file_path),
            line_number=line_no,
            code_snippet=code_snippet.strip(),
            remediation=remediation,
            confidence="MEDIUM",
        )

        self._enrich_finding_metadata(finding)
        self.findings.append(finding)

    def _enrich_finding_metadata(self, finding: SecurityFinding) -> None:
        """Enrich finding with CWE, OWASP, and other metadata."""
        # CWE mapping
        cwe_mapping = {
            VulnerabilityType.SQL_INJECTION: "CWE-89",
            VulnerabilityType.XSS: "CWE-79",
            VulnerabilityType.COMMAND_INJECTION: "CWE-78",
            VulnerabilityType.PATH_TRAVERSAL: "CWE-22",
            VulnerabilityType.DESERIALIZATION: "CWE-502",
            VulnerabilityType.CRYPTO_WEAKNESS: "CWE-327",
            VulnerabilityType.HARDCODED_SECRET: "CWE-798",
            VulnerabilityType.DANGEROUS_FUNCTION: "CWE-94",
            VulnerabilityType.INPUT_VALIDATION: "CWE-20",
            VulnerabilityType.RACE_CONDITION: "CWE-362",
        }

        # OWASP Top 10 mapping
        owasp_mapping = {
            VulnerabilityType.SQL_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.XSS: "A03:2021 – Injection",
            VulnerabilityType.COMMAND_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.AUTH_BYPASS: "A07:2021 – Identification and Authentication Failures",
            VulnerabilityType.CRYPTO_WEAKNESS: "A02:2021 – Cryptographic Failures",
            VulnerabilityType.HARDCODED_SECRET: "A02:2021 – Cryptographic Failures",
            VulnerabilityType.INPUT_VALIDATION: "A03:2021 – Injection",
            VulnerabilityType.INFO_DISCLOSURE: "A01:2021 – Broken Access Control",
        }

        finding.cwe_id = cwe_mapping.get(finding.vulnerability_type)
        finding.owasp_category = owasp_mapping.get(finding.vulnerability_type)

        # Add references
        if finding.cwe_id:
            finding.references.append(
                f"https://cwe.mitre.org/data/definitions/{finding.cwe_id[4:]}.html"
            )

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        if self.exclude_test_files and "test" in str(file_path).lower():
            return True

        # Skip common non-security-relevant files
        skip_patterns = [
            "__pycache__",
            ".pyc",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".tox",
            "build",
            "dist",
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _map_bandit_test_to_vuln_type(self, test_id: str) -> VulnerabilityType:
        """Map bandit test ID to vulnerability type."""
        mapping = {
            "B102": VulnerabilityType.DANGEROUS_FUNCTION,  # exec_used
            "B301": VulnerabilityType.DANGEROUS_FUNCTION,  # pickle
            "B506": VulnerabilityType.DANGEROUS_FUNCTION,  # yaml_load
            "B602": VulnerabilityType.COMMAND_INJECTION,  # subprocess_popen_with_shell_equals_true
            "B608": VulnerabilityType.SQL_INJECTION,  # hardcoded_sql_expressions
            "B703": VulnerabilityType.CRYPTO_WEAKNESS,  # django_mark_safe
        }
        return mapping.get(test_id, VulnerabilityType.INPUT_VALIDATION)

    def _map_safety_severity(self, vuln_id: str) -> Severity:
        """Map safety vulnerability ID to severity."""
        # This would typically query a CVE database
        # For now, return medium severity
        return Severity.MEDIUM


class SecurityReportGenerator:
    """Generate security reports in various formats."""

    @staticmethod
    def generate_console_report(report: SecurityReport) -> str:
        """Generate colorized console report."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("🔐 PYTHON SECURITY AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"📁 Target: {report.target_path}")
        lines.append(
            f"🕒 Scan time: {report.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"⏱️  Duration: {report.scan_duration:.2f} seconds")
        lines.append("")

        # Summary
        lines.append("📊 SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Security Score: {report.summary['security_score']}/100")
        lines.append(f"Total Findings: {report.summary['total_findings']}")
        lines.append(f"Files Scanned: {report.summary['files_scanned']}")
        lines.append(
            f"Dependency Vulnerabilities: {report.summary['dependency_vulnerabilities']}"
        )
        lines.append("")

        # Severity breakdown
        lines.append("🚨 SEVERITY BREAKDOWN")
        lines.append("-" * 40)
        for severity, count in report.summary["severity_breakdown"].items():
            if count > 0:
                severity_emoji = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🟢",
                    "INFO": "🔵",
                }
                lines.append(
                    f"{severity_emoji.get(severity, '⚪')} {severity}: {count}"
                )
        lines.append("")

        # Top vulnerability types
        if report.summary["vulnerability_types"]:
            lines.append("🎯 TOP VULNERABILITY TYPES")
            lines.append("-" * 40)
            sorted_vulns = sorted(
                report.summary["vulnerability_types"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for vuln_type, count in sorted_vulns:
                lines.append(f"• {vuln_type.replace('_', ' ').title()}: {count}")
            lines.append("")

        # Critical and High findings
        critical_high = [
            f
            for f in report.findings
            if f.severity in [Severity.CRITICAL, Severity.HIGH]
        ]

        if critical_high:
            lines.append("🚨 CRITICAL & HIGH SEVERITY FINDINGS")
            lines.append("-" * 40)

            for finding in critical_high[:10]:  # Show top 10
                lines.append(f"🔴 {finding.title}")
                lines.append(f"   📁 {finding.file_path}:{finding.line_number}")
                lines.append(f"   📝 {finding.description}")
                if finding.remediation:
                    lines.append(f"   🔧 {finding.remediation}")
                lines.append("")

        # Dependency vulnerabilities
        if report.dependency_vulnerabilities:
            lines.append("📦 DEPENDENCY VULNERABILITIES")
            lines.append("-" * 40)
            for dep_vuln in report.dependency_vulnerabilities[:5]:  # Show top 5
                lines.append(f"• {dep_vuln.package} {dep_vuln.version}")
                lines.append(f"  ID: {dep_vuln.vulnerability_id}")
                lines.append(f"  Advisory: {dep_vuln.advisory}")
                lines.append("")

        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)

        if report.summary["critical_high_issues"] > 0:
            lines.append("🚨 Address critical and high severity issues immediately")

        if report.summary["dependency_vulnerabilities"] > 0:
            lines.append("📦 Update vulnerable dependencies")

        if report.false_positives > 0:
            lines.append(
                f"🎯 Review {report.false_positives} potential false positives"
            )

        lines.append("🔍 Integrate security scanning into CI/CD pipeline")
        lines.append("📚 Provide security training for development team")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def generate_json_report(report: SecurityReport) -> str:
        """Generate JSON report for programmatic processing."""
        report_dict = {
            "metadata": {
                "scan_timestamp": report.scan_timestamp.isoformat(),
                "target_path": report.target_path,
                "scan_duration": report.scan_duration,
                "tool_version": "1.0.0",
                "format_version": "1.0",
            },
            "summary": report.summary,
            "metrics": report.metrics,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity.value,
                    "vulnerability_type": f.vulnerability_type.value,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "column": f.column,
                    "code_snippet": f.code_snippet,
                    "remediation": f.remediation,
                    "references": f.references,
                    "cwe_id": f.cwe_id,
                    "owasp_category": f.owasp_category,
                    "cvss_score": f.cvss_score,
                    "confidence": f.confidence,
                    "test_id": f.test_id,
                    "false_positive": f.false_positive,
                }
                for f in report.findings
            ],
            "dependency_vulnerabilities": [
                {
                    "package": dv.package,
                    "version": dv.version,
                    "vulnerability_id": dv.vulnerability_id,
                    "advisory": dv.advisory,
                    "affected_versions": dv.affected_versions,
                    "fixed_version": dv.fixed_version,
                    "severity": dv.severity.value,
                }
                for dv in report.dependency_vulnerabilities
            ],
        }

        return json.dumps(report_dict, indent=2, sort_keys=True)


def main():
    """Main entry point for the security auditor."""
    import argparse

    parser = argparse.ArgumentParser(description="Python Security Auditor")
    parser.add_argument("target_path", help="Path to analyze")
    parser.add_argument(
        "--format", choices=["console", "json"], default="console", help="Output format"
    )
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--include-low", action="store_true", help="Include low severity findings"
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        default=True,
        help="Exclude test files from analysis",
    )
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)

    config.update(
        {
            "include_low_severity": args.include_low,
            "exclude_test_files": args.exclude_tests,
        }
    )

    # Run security audit
    auditor = PythonSecurityAuditor(args.target_path, config)
    report = auditor.audit()

    # Generate report
    if args.format == "json":
        output = SecurityReportGenerator.generate_json_report(report)
    else:
        output = SecurityReportGenerator.generate_console_report(report)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
