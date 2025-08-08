#!/usr/bin/env python3
"""
Security Audit Task Launcher

A demonstration script showing how to launch the Python Security Auditor agent
via the Task tool interface for comprehensive security analysis.

This script demonstrates various ways to invoke the security auditor:
1. Basic security audit of the current project
2. Comprehensive audit with detailed reporting
3. CI/CD integration example with exit codes
4. Custom configuration examples

Usage:
    python launch_security_audit.py [mode]

Modes:
    basic       - Basic security audit with console output
    detailed    - Detailed audit with JSON report generation
    ci          - CI/CD mode with structured output and exit codes
    demo        - Demonstration of all features
"""

import sys
import json
from pathlib import Path
from task_agents.python_security_auditor_agent import run_security_audit_agent


def basic_audit():
    """Run a basic security audit with console output."""
    print("🔍 Running Basic Security Audit")
    print("=" * 50)

    try:
        result = run_security_audit_agent(
            target_path="./components",
            include_low_severity=False,
            exclude_test_files=True,
            output_format="console",
        )

        print("\n✅ Audit completed successfully!")
        print(f"Security Score: {result['security_score']}/100")
        print(f"Issues Found: {result['total_findings']}")

        return result["critical_high_issues"] == 0

    except Exception as e:
        print(f"❌ Audit failed: {e}")
        return False


def detailed_audit():
    """Run a detailed security audit with JSON report generation."""
    print("🔍 Running Detailed Security Audit")
    print("=" * 50)

    try:
        result = run_security_audit_agent(
            target_path=".",
            include_low_severity=True,
            exclude_test_files=False,  # Include test files for comprehensive analysis
            output_format="json",
            output_file="comprehensive_security_report.json",
            max_file_size=2 * 1024 * 1024,  # 2MB file size limit
        )

        print("\n📊 Detailed Analysis Results:")
        print(f"Security Score: {result['security_score']}/100")
        print(f"Total Findings: {result['total_findings']}")
        print(f"Critical/High Issues: {result['critical_high_issues']}")
        print(f"Dependency Vulnerabilities: {result['dependency_vulnerabilities']}")
        print(f"Scan Duration: {result['scan_duration']:.2f} seconds")

        # Show top findings
        if result["top_findings_summary"]:
            print("\n🚨 Top Security Issues:")
            for i, finding in enumerate(result["top_findings_summary"][:3], 1):
                print(f"  {i}. [{finding['severity']}] {finding['title']}")
                print(f"     File: {finding['file']}:{finding['line']}")
                print(f"     Fix: {finding['remediation'][:80]}...")

        # Show dependency issues
        if result["dependency_summary"]:
            print("\n📦 Dependency Vulnerabilities:")
            for i, dep in enumerate(result["dependency_summary"][:3], 1):
                print(f"  {i}. {dep['package']} {dep['version']}")
                print(f"     Vulnerability: {dep['vulnerability_id']}")

        print(f"\n📄 Full report saved to: {result['report_file']}")

        return result["critical_high_issues"] == 0

    except Exception as e:
        print(f"❌ Detailed audit failed: {e}")
        return False


def ci_cd_audit():
    """Run security audit in CI/CD mode with structured output and exit codes."""
    print("🔍 Running CI/CD Security Audit")
    print("=" * 50)

    try:
        result = run_security_audit_agent(
            target_path="./components",
            include_low_severity=False,
            exclude_test_files=True,
            output_format="json",
            output_file="ci_security_report.json",
        )

        # CI/CD structured output
        ci_summary = {
            "status": "SUCCESS" if result["critical_high_issues"] == 0 else "FAILURE",
            "security_score": result["security_score"],
            "total_findings": result["total_findings"],
            "critical_high_issues": result["critical_high_issues"],
            "scan_duration": result["scan_duration"],
            "report_file": result["report_file"],
        }

        print(json.dumps(ci_summary, indent=2))

        # Exit with appropriate code for CI/CD
        if result["critical_high_issues"] > 0:
            print(
                f"\n❌ Security gate failed: {result['critical_high_issues']} critical/high issues found"
            )
            return False
        else:
            print("\n✅ Security gate passed: No critical/high issues found")
            return True

    except Exception as e:
        print(f"❌ CI/CD audit failed: {e}")
        print(json.dumps({"status": "ERROR", "error": str(e)}, indent=2))
        return False


def demo_audit():
    """Demonstrate all features of the security auditor agent."""
    print("🔍 Security Auditor Agent Demonstration")
    print("=" * 60)

    # Demo 1: Basic audit
    print("\n1️⃣  BASIC AUDIT DEMO")
    print("-" * 30)
    basic_result = basic_audit()

    # Demo 2: Custom configuration audit
    print("\n\n2️⃣  CUSTOM CONFIGURATION DEMO")
    print("-" * 30)

    try:
        custom_config = {
            "include_low_severity": True,
            "exclude_test_files": False,
            "whitelist_files": ["legacy_code.py"],
            "custom_patterns": {"DEMO_PATTERN": ["suspicious_function"]},
        }

        result = run_security_audit_agent(
            target_path="./components/sejm_whiz",
            **custom_config,
            output_format="console",
        )

        print(f"Custom audit completed - Score: {result['security_score']}/100")

    except Exception as e:
        print(f"Custom audit failed: {e}")

    # Demo 3: Report format comparison
    print("\n\n3️⃣  REPORT FORMAT COMPARISON")
    print("-" * 30)

    try:
        # JSON report
        run_security_audit_agent(
            target_path="./python_security_auditor.py",
            output_format="json",
            output_file="demo_json_report.json",
        )

        # Console report
        run_security_audit_agent(
            target_path="./python_security_auditor.py", output_format="console"
        )

        print("Reports generated successfully!")
        print("JSON report: demo_json_report.json")
        print("Console report displayed above")

    except Exception as e:
        print(f"Report demo failed: {e}")

    # Demo 4: Performance metrics
    print("\n\n4️⃣  PERFORMANCE METRICS DEMO")
    print("-" * 30)

    try:
        perf_result = run_security_audit_agent(
            target_path=".", include_low_severity=False, exclude_test_files=True
        )

        metrics = perf_result["metrics"]
        print(f"Scan Duration: {metrics['scan_duration']:.2f} seconds")
        print(f"Files per Second: {metrics['files_per_second']:.1f}")
        print(f"Findings per File: {metrics['findings_per_file']:.2f}")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.1%}")

    except Exception as e:
        print(f"Performance demo failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION COMPLETE")
    print("✅ The Python Security Auditor Agent is ready for use!")
    print("📚 See SECURITY_AUDITOR_README.md for full documentation")

    return basic_result


def main():
    """Main entry point for the security audit launcher."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    modes = {
        "basic": basic_audit,
        "detailed": detailed_audit,
        "ci": ci_cd_audit,
        "demo": demo_audit,
    }

    if mode not in modes:
        print(f"❌ Unknown mode: {mode}")
        print(f"Available modes: {', '.join(modes.keys())}")
        sys.exit(1)

    print(f"🚀 Launching Security Audit in '{mode}' mode")
    print(f"📁 Working directory: {Path.cwd()}")

    try:
        success = modes[mode]()

        if success:
            print("\n🎉 Security audit completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Security audit completed with issues found")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Security audit interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Security audit failed with error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
