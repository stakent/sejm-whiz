# Python Security Auditor Agent

A comprehensive security analysis tool for Python codebases that identifies vulnerabilities, security anti-patterns, and provides actionable remediation guidance.

## Overview

The Python Security Auditor Agent is a sophisticated security scanning tool that combines multiple analysis techniques to identify security vulnerabilities in Python code. It integrates with industry-standard tools like Bandit and Safety while providing custom pattern detection and advanced security analysis capabilities.

## Features

### Core Security Analysis

- **Static Security Analysis**: Integration with Bandit for industry-standard vulnerability detection
- **Dependency Vulnerability Scanning**: Integration with Safety for known CVE detection in dependencies
- **Custom Pattern Detection**: Advanced regex-based pattern matching for security anti-patterns
- **AST-Based Analysis**: Abstract Syntax Tree analysis for dangerous function usage
- **Taint Analysis**: Basic data flow analysis to track user input through code paths
- **Configuration Security**: Analysis of configuration files for security misconfigurations

### Vulnerability Detection Categories

- **SQL Injection**: Detection of unsafe query construction and ORM misuse
- **Cross-Site Scripting (XSS)**: Template injection and unsafe output patterns
- **Command Injection**: Unsafe subprocess and system command usage
- **Path Traversal**: Directory traversal vulnerabilities in file operations
- **Deserialization Attacks**: Unsafe pickle, eval, and exec usage
- **Authentication/Authorization Flaws**: Access control bypasses
- **Cryptographic Weaknesses**: Weak algorithms and hardcoded secrets
- **Input Validation Issues**: Insufficient user input sanitization
- **Race Conditions**: TOCTOU (Time-of-Check-Time-of-Use) vulnerabilities
- **Supply Chain Security**: Unpinned dependencies and package vulnerabilities

### Advanced Analysis Features

- **False Positive Reduction**: Context-aware analysis to minimize false alarms
- **Severity Classification**: CRITICAL, HIGH, MEDIUM, LOW, INFO severity levels
- **OWASP Top 10 Mapping**: Integration with OWASP security categories
- **CWE Classification**: Common Weakness Enumeration mapping
- **Confidence Scoring**: High, Medium, Low confidence ratings
- **Remediation Guidance**: Actionable fix recommendations for each finding

### Reporting Capabilities

- **Console Output**: Colorized, human-readable reports with severity indicators
- **JSON Format**: Structured data for CI/CD integration and programmatic processing
- **Security Scoring**: Overall security posture scoring (0-100 scale)
- **Executive Summary**: High-level metrics and key findings
- **Detailed Findings**: Complete vulnerability details with code snippets and remediation

## Installation

The security auditor requires the following Python packages for full functionality:

```bash
# Install core dependencies
pip install bandit safety

# Optional: For enhanced analysis capabilities
pip install ast-monitor  # For advanced AST analysis
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Analyze current directory
python python_security_auditor.py .

# Analyze specific directory
python python_security_auditor.py ./components

# Include low severity findings
python python_security_auditor.py ./src --include-low

# Generate JSON report
python python_security_auditor.py ./src --format json --output security_report.json
```

#### Advanced Options

```bash
# Full feature analysis with custom configuration
python python_security_auditor.py ./codebase \
    --format json \
    --output detailed_security_report.json \
    --include-low \
    --config security_config.json
```

### Task Agent Interface

The agent can also be launched via the task agent wrapper for integration with larger workflows:

```bash
# Basic security audit
python task_agents/python_security_auditor_agent.py ./components

# Advanced audit with custom options
python task_agents/python_security_auditor_agent.py ./src \
    --format json \
    --output security_report.json \
    --include-low \
    --no-exclude-tests \
    --quiet
```

### Programmatic Usage

```python
from python_security_auditor import PythonSecurityAuditor, SecurityReportGenerator

# Initialize auditor with configuration
config = {
    'include_low_severity': True,
    'exclude_test_files': True,
    'custom_patterns': {},
    'whitelist_files': [],
    'max_file_size': 1024 * 1024  # 1MB
}

auditor = PythonSecurityAuditor('./my_project', config)
report = auditor.audit()

# Generate different report formats
console_report = SecurityReportGenerator.generate_console_report(report)
json_report = SecurityReportGenerator.generate_json_report(report)

print(f"Security Score: {report.summary['security_score']}/100")
print(f"Critical/High Issues: {report.summary['critical_high_issues']}")
```

## Configuration

### Configuration File Format

Create a JSON configuration file to customize the security audit:

```json
{
    "include_low_severity": true,
    "exclude_test_files": true,
    "max_file_size": 1048576,
    "custom_patterns": {
        "additional_dangerous_functions": ["dangerous_func1", "dangerous_func2"]
    },
    "whitelist_files": [
        "legacy_code.py",
        "third_party_integration.py"
    ]
}
```

### Configuration Options

- **include_low_severity** (bool): Include low severity findings in the report
- **exclude_test_files** (bool): Skip test files during analysis
- **max_file_size** (int): Maximum file size to analyze in bytes (default: 1MB)
- **custom_patterns** (dict): Additional custom security patterns
- **whitelist_files** (list): File patterns to exclude from analysis

## Output Formats

### Console Report

```
================================================================================
🔐 PYTHON SECURITY AUDIT REPORT
================================================================================
📁 Target: ./components
🕒 Scan time: 2025-01-13 10:30:45
⏱️  Duration: 2.34 seconds

📊 SUMMARY
----------------------------------------
Security Score: 85/100
Total Findings: 12
Files Scanned: 47
Dependency Vulnerabilities: 3

🚨 SEVERITY BREAKDOWN
----------------------------------------
🔴 CRITICAL: 1
🟠 HIGH: 4
🟡 MEDIUM: 5
🟢 LOW: 2

🚨 CRITICAL & HIGH SEVERITY FINDINGS
----------------------------------------
🔴 Use of eval() can lead to arbitrary code execution
   📁 ./components/parser.py:45
   📝 Use of eval() can lead to arbitrary code execution
   🔧 Replace eval() with ast.literal_eval() for safe evaluation
```

### JSON Report Structure

```json
{
    "metadata": {
        "scan_timestamp": "2025-01-13T10:30:45",
        "target_path": "./components",
        "scan_duration": 2.34,
        "tool_version": "1.0.0",
        "format_version": "1.0"
    },
    "summary": {
        "total_findings": 12,
        "security_score": 85,
        "critical_high_issues": 5,
        "severity_breakdown": {
            "CRITICAL": 1,
            "HIGH": 4,
            "MEDIUM": 5,
            "LOW": 2,
            "INFO": 0
        }
    },
    "findings": [
        {
            "id": "BANDIT_B102_1234567890",
            "severity": "CRITICAL",
            "vulnerability_type": "DANGEROUS_FUNCTION",
            "title": "Use of eval() detected",
            "description": "Use of eval() can lead to arbitrary code execution",
            "file_path": "./components/parser.py",
            "line_number": 45,
            "code_snippet": "result = eval(user_input)",
            "remediation": "Replace eval() with ast.literal_eval()",
            "cwe_id": "CWE-94",
            "owasp_category": "A03:2021 – Injection",
            "confidence": "HIGH"
        }
    ]
}
```

## Security Patterns Detected

### Dangerous Functions

- **eval()**: Arbitrary code execution risk
- **exec()**: Dynamic code execution vulnerability
- **compile()**: Dynamic compilation security issues
- **__import__()**: Module injection vulnerabilities

### Injection Vulnerabilities

- **SQL Injection**: String formatting and concatenation in SQL queries
- **Command Injection**: Unsafe subprocess and os.system usage
- **XSS**: Template injection in web frameworks
- **Path Traversal**: Directory traversal in file operations

### Cryptographic Issues

- **Weak Algorithms**: MD5, SHA1, DES, RC4 usage
- **Hardcoded Secrets**: API keys, passwords, tokens in source code
- **Insecure Random**: Non-cryptographic random number generation

### Configuration Security

- **Debug Mode**: Production debug flags enabled
- **Empty Allowed Hosts**: Django security misconfigurations
- **Disabled SSL Verification**: Certificate validation bypasses

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Security Audit
on: [push, pull_request]

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install bandit safety

      - name: Run security audit
        run: |
          python python_security_auditor.py . \
            --format json \
            --output security_report.json

      - name: Upload security report
        uses: actions/upload-artifact@v3
        with:
          name: security-report
          path: security_report.json

      - name: Check for critical issues
        run: |
          # Fail build if critical or high severity issues found
          python -c "
          import json
          with open('security_report.json') as f:
              report = json.load(f)
          critical_high = report['summary']['critical_high_issues']
          if critical_high > 0:
              print(f'❌ Found {critical_high} critical/high severity security issues')
              exit(1)
          print('✅ No critical/high severity security issues found')
          "
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: security-audit
        name: Python Security Audit
        entry: python python_security_auditor.py
        language: system
        args: ['.', '--format', 'console']
        pass_filenames: false
```

## Best Practices

### Regular Scanning

- Run security audits on every commit/push
- Schedule weekly comprehensive scans
- Monitor for new vulnerability disclosures

### Remediation Workflow

1. **Prioritize Critical/High Issues**: Address immediately
1. **Review Medium Issues**: Evaluate in context
1. **Plan Low/Info Issues**: Include in technical debt backlog
1. **Update Dependencies**: Keep packages current
1. **Validate Fixes**: Re-scan after remediation

### False Positive Management

- Use configuration to whitelist known safe patterns
- Document security exceptions with justification
- Review whitelist regularly for relevance

### Team Integration

- Share security reports with development teams
- Provide security training based on common findings
- Establish security review processes for critical components

## Limitations and Considerations

### Analysis Scope

- Static analysis cannot detect runtime-specific vulnerabilities
- Context-dependent security issues may require manual review
- Custom business logic vulnerabilities need domain expertise

### Performance Considerations

- Large codebases may require extended scan times
- File size limits prevent analysis of very large files
- Memory usage scales with codebase complexity

### Tool Integration

- Bandit and Safety must be installed for full functionality
- Some advanced features require additional Python packages
- Integration with specific frameworks may need customization

## Support and Contribution

### Extending the Auditor

The security auditor is designed for extensibility:

```python
# Add custom security patterns
custom_patterns = {
    'CUSTOM_DANGEROUS_FUNCTIONS': {
        'risky_function': {
            'severity': Severity.HIGH,
            'message': 'Custom security warning',
            'remediation': 'Use safer alternative'
        }
    }
}

auditor = PythonSecurityAuditor('./code', {
    'custom_patterns': custom_patterns
})
```

### Contributing Security Rules

- Add new vulnerability patterns to SecurityPatterns class
- Enhance CWE and OWASP mappings for better classification
- Improve false positive detection with context rules
- Add support for additional security tools and frameworks

This security auditor agent provides comprehensive security analysis capabilities for Python codebases, helping teams identify and address security vulnerabilities proactively. Regular use of this tool as part of your development workflow significantly improves your application's security posture.
