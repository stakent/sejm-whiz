#!/usr/bin/env python3
"""
Task Agent: Python Security Auditor

A task agent wrapper for the comprehensive Python security analysis tool.
This agent can be launched via the Task tool to perform security audits.

Usage:
    python task_agents/python_security_auditor_agent.py <target_path> [options]

Example:
    python task_agents/python_security_auditor_agent.py ./components --format json --output security_report.json
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import the main auditor
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_security_auditor import PythonSecurityAuditor, SecurityReportGenerator


def run_security_audit_agent(target_path: str, **kwargs) -> dict:
    """
    Run the Python Security Auditor agent.

    Args:
        target_path: Path to the Python codebase to analyze
        **kwargs: Additional configuration options
            - include_low_severity: Include low severity findings (default: True)
            - exclude_test_files: Exclude test files from analysis (default: True)
            - output_format: Output format ('console' or 'json', default: 'console')
            - output_file: Output file path (optional)
            - custom_patterns: Dictionary of custom security patterns (optional)
            - whitelist_files: List of file patterns to whitelist (optional)
            - max_file_size: Maximum file size to analyze in bytes (default: 1MB)

    Returns:
        dict: Comprehensive security analysis results
    """

    print("🚀 Launching Python Security Auditor Agent")
    print(f"📍 Target: {target_path}")

    # Validate target path
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Target path does not exist: {target_path}")

    # Prepare configuration
    config = {
        "include_low_severity": kwargs.get("include_low_severity", True),
        "exclude_test_files": kwargs.get("exclude_test_files", True),
        "custom_patterns": kwargs.get("custom_patterns", {}),
        "whitelist_files": kwargs.get("whitelist_files", []),
        "max_file_size": kwargs.get("max_file_size", 1024 * 1024),  # 1MB
    }

    print(f"⚙️  Configuration: {config}")

    # Initialize and run the security auditor
    auditor = PythonSecurityAuditor(target_path, config)
    report = auditor.audit()

    # Generate output in requested format
    output_format = kwargs.get("output_format", "console")
    output_file = kwargs.get("output_file")

    if output_format == "json":
        output_content = SecurityReportGenerator.generate_json_report(report)
    else:
        output_content = SecurityReportGenerator.generate_console_report(report)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_content)
        print(f"📄 Report saved to: {output_file}")

    # Return structured results for programmatic access
    result = {
        "success": True,
        "scan_timestamp": report.scan_timestamp.isoformat(),
        "target_path": report.target_path,
        "summary": report.summary,
        "metrics": report.metrics,
        "total_findings": len(report.findings),
        "dependency_vulnerabilities": len(report.dependency_vulnerabilities),
        "security_score": report.summary["security_score"],
        "critical_high_issues": report.summary["critical_high_issues"],
        "scan_duration": report.scan_duration,
        "output_content": output_content,
        "report_file": output_file,
    }

    # Add top findings summary for quick review
    critical_high_findings = [
        f for f in report.findings if f.severity.value in ["CRITICAL", "HIGH"]
    ]

    result["top_findings_summary"] = []
    for finding in critical_high_findings[:10]:  # Top 10 critical/high issues
        result["top_findings_summary"].append(
            {
                "severity": finding.severity.value,
                "type": finding.vulnerability_type.value,
                "title": finding.title,
                "file": finding.file_path,
                "line": finding.line_number,
                "remediation": finding.remediation,
            }
        )

    # Add dependency vulnerability summary
    result["dependency_summary"] = []
    for dep_vuln in report.dependency_vulnerabilities[:5]:  # Top 5 dependency issues
        result["dependency_summary"].append(
            {
                "package": dep_vuln.package,
                "version": dep_vuln.version,
                "vulnerability_id": dep_vuln.vulnerability_id,
                "severity": dep_vuln.severity.value,
                "advisory": dep_vuln.advisory[:100] + "..."
                if len(dep_vuln.advisory) > 100
                else dep_vuln.advisory,
            }
        )

    return result


def main():
    """Command-line interface for the agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Python Security Auditor Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task_agents/python_security_auditor_agent.py ./components
  python task_agents/python_security_auditor_agent.py ./src --format json --output report.json
  python task_agents/python_security_auditor_agent.py . --include-low --no-exclude-tests
        """,
    )

    parser.add_argument("target_path", help="Path to analyze")
    parser.add_argument(
        "--format",
        choices=["console", "json"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument("--output", help="Output file path (default: print to console)")
    parser.add_argument(
        "--include-low", action="store_true", help="Include low severity findings"
    )
    parser.add_argument(
        "--no-exclude-tests", action="store_true", help="Include test files in analysis"
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1024 * 1024,
        help="Maximum file size to analyze in bytes (default: 1MB)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output, only show final results",
    )

    args = parser.parse_args()

    try:
        # Configure options
        options = {
            "include_low_severity": args.include_low,
            "exclude_test_files": not args.no_exclude_tests,
            "output_format": args.format,
            "output_file": args.output,
            "max_file_size": args.max_file_size,
        }

        # Run the agent
        if args.quiet:
            # Temporarily redirect stdout to suppress progress output
            import io
            import contextlib

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = run_security_audit_agent(args.target_path, **options)
        else:
            result = run_security_audit_agent(args.target_path, **options)

        # Print summary for command line usage
        if not args.output or not args.quiet:
            print("\n" + "=" * 60)
            print("🔐 SECURITY AUDIT SUMMARY")
            print("=" * 60)
            print(f"✅ Security Score: {result['security_score']}/100")
            print(f"🚨 Total Findings: {result['total_findings']}")
            print(f"⚠️  Critical/High Issues: {result['critical_high_issues']}")
            print(
                f"📦 Dependency Vulnerabilities: {result['dependency_vulnerabilities']}"
            )
            print(f"⏱️  Scan Duration: {result['scan_duration']:.2f}s")

            if result["critical_high_issues"] > 0:
                print("\n🚨 TOP CRITICAL/HIGH FINDINGS:")
                for i, finding in enumerate(result["top_findings_summary"][:5], 1):
                    print(f"  {i}. [{finding['severity']}] {finding['title']}")
                    print(f"     📁 {finding['file']}:{finding['line']}")
                    if finding["remediation"]:
                        print(f"     🔧 {finding['remediation'][:100]}...")

            if result["dependency_vulnerabilities"] > 0:
                print("\n📦 TOP DEPENDENCY VULNERABILITIES:")
                for i, dep in enumerate(result["dependency_summary"][:3], 1):
                    print(f"  {i}. {dep['package']} {dep['version']}")
                    print(f"     🆔 {dep['vulnerability_id']} [{dep['severity']}]")

            print("=" * 60)

        # Return exit code based on findings
        if result["critical_high_issues"] > 0:
            sys.exit(1)  # Exit with error if critical/high issues found
        else:
            sys.exit(0)  # Success

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
