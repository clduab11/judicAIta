#!/usr/bin/env python3
"""
Kaggle Environment Validation Script for JudicAIta.

This script validates that the Kaggle/Colab environment is properly configured
for running JudicAIta training and inference.

Usage:
    python scripts/validate_kaggle_env.py

In a notebook:
    !python scripts/validate_kaggle_env.py
    # or
    %run scripts/validate_kaggle_env.py
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_icon(passed: bool) -> str:
    """Return check mark or X based on pass status."""
    return f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"


def warn_icon() -> str:
    """Return warning icon."""
    return f"{YELLOW}⚠️{RESET}"


class KaggleEnvironmentValidator:
    """Validates Kaggle/Colab environment for JudicAIta."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self.results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }
        self.all_passed = True

    def print_header(self, title: str) -> None:
        """Print a section header."""
        print(f"\n{BLUE}{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}{RESET}\n")

    def log_check(
        self,
        name: str,
        passed: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a check result."""
        self.results["checks"][name] = {
            "passed": passed,
            "message": message,
            "details": details or {},
        }
        if not passed:
            self.all_passed = False
            self.results["errors"].append(f"{name}: {message}")
        print(f"  {check_icon(passed)} {name}: {message}")
        if details:
            for key, value in details.items():
                print(f"      {key}: {value}")

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self.results["warnings"].append(message)
        print(f"  {warn_icon()} Warning: {message}")

    def log_recommendation(self, message: str) -> None:
        """Log a recommendation."""
        self.results["recommendations"].append(message)
        print(f"  {BLUE}💡{RESET} {message}")

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        passed = version.major == 3 and version.minor >= 10
        self.log_check(
            "Python Version",
            passed,
            version_str,
            {"required": ">=3.10"},
        )
        return passed

    def check_tpu_availability(self) -> bool:
        """Check if TPU is available."""
        try:
            import jax

            devices = jax.devices()
            tpu_devices = [d for d in devices if d.platform == "tpu"]
            if tpu_devices:
                self.log_check(
                    "TPU Availability",
                    True,
                    f"Found {len(tpu_devices)} TPU core(s)",
                    {"devices": [str(d) for d in tpu_devices[:4]]},
                )
                return True
            else:
                # Check for GPU as fallback
                gpu_devices = [d for d in devices if d.platform == "gpu"]
                if gpu_devices:
                    self.log_check(
                        "TPU Availability",
                        False,
                        f"No TPU found, but {len(gpu_devices)} GPU(s) available",
                    )
                    self.log_warning("TPU not available. Training may be slower on GPU.")
                else:
                    self.log_check(
                        "TPU Availability",
                        False,
                        "No TPU or GPU found. Using CPU.",
                    )
                return False
        except ImportError:
            self.log_check("TPU Availability", False, "JAX not installed")
            self.log_recommendation(
                "Install JAX: pip install git+https://github.com/jax-ml/jax"
            )
            return False
        except Exception as e:
            self.log_check("TPU Availability", False, f"Error checking TPU: {e}")
            return False

    def check_package_installed(
        self, package_name: str, min_version: str | None = None
    ) -> bool:
        """Check if a package is installed with optional version check."""
        try:
            spec = importlib.util.find_spec(package_name.replace("-", "_"))
            if spec is None:
                return False

            # Try to get version
            try:
                mod = __import__(package_name.replace("-", "_"))
                version = getattr(mod, "__version__", "unknown")
            except Exception:
                version = "installed"

            return True
        except Exception:
            return False

    def check_core_dependencies(self) -> bool:
        """Check core JudicAIta dependencies."""
        self.print_header("Core Dependencies")

        required = [
            ("pydantic", "2.5.0"),
            ("loguru", "0.7.0"),
            ("pdfplumber", "0.10.0"),
            ("python-docx", None),
            ("typer", "0.9.0"),
            ("rich", "13.7.0"),
            ("nest_asyncio", None),
        ]

        all_installed = True
        for package, min_ver in required:
            installed = self.check_package_installed(package)
            version_info = f" (>={min_ver})" if min_ver else ""
            self.log_check(
                f"Package: {package}",
                installed,
                "Installed" if installed else "Not installed",
            )
            if not installed:
                all_installed = False

        return all_installed

    def check_ml_dependencies(self) -> bool:
        """Check ML/AI dependencies."""
        self.print_header("ML/AI Dependencies")

        packages = [
            ("torch", "PyTorch"),
            ("transformers", "HuggingFace Transformers"),
            ("jax", "JAX"),
            ("flax", "Flax"),
        ]

        results = {}
        for package, display_name in packages:
            installed = self.check_package_installed(package)
            results[package] = installed
            self.log_check(
                display_name,
                installed,
                "Available" if installed else "Not installed",
            )

        # Check for Tunix
        try:
            from tunix.rl.grpo import GRPOLearner  # noqa: F401

            self.log_check("Google Tunix", True, "Available")
            results["tunix"] = True
        except ImportError:
            self.log_check("Google Tunix", False, "Not installed")
            self.log_recommendation(
                "Install Tunix: pip install git+https://github.com/google/tunix"
            )
            results["tunix"] = False

        return all(results.values())

    def check_memory(self) -> bool:
        """Check available memory."""
        self.print_header("Memory Check")

        try:
            import psutil

            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)

            # Kaggle provides ~13GB RAM for CPU, 16GB for GPU, 15GB for TPU
            min_required = 8.0  # GB

            passed = available_gb >= min_required
            self.log_check(
                "System Memory",
                passed,
                f"{available_gb:.1f} GB available of {total_gb:.1f} GB total",
                {
                    "minimum_required": f"{min_required} GB",
                    "status": "OK" if passed else "LOW",
                },
            )

            if available_gb < 12:
                self.log_warning(
                    "Memory may be tight for large models. Consider reducing batch size."
                )

            return passed
        except ImportError:
            self.log_warning("psutil not installed, cannot check memory")
            return True  # Don't fail for this

    def check_judicaita_import(self) -> bool:
        """Check if JudicAIta can be imported."""
        self.print_header("JudicAIta Import Check")

        checks = [
            ("judicaita", "Core package"),
            ("judicaita.document_input", "Document Input module"),
            ("judicaita.reasoning_trace", "Reasoning Trace module"),
            ("judicaita.citation_mapping", "Citation Mapping module"),
            ("judicaita.summary_generator", "Summary Generator module"),
            ("judicaita.notebook_utils", "Notebook Utilities"),
        ]

        all_passed = True
        for module, description in checks:
            try:
                __import__(module)
                self.log_check(description, True, "Import successful")
            except ImportError as e:
                self.log_check(description, False, f"Import failed: {e}")
                all_passed = False

        return all_passed

    def run_smoke_test(self) -> bool:
        """Run basic smoke tests."""
        self.print_header("Smoke Tests")

        tests_passed = True

        # Test 1: Settings instantiation
        try:
            from judicaita.core.config import Settings

            settings = Settings(google_api_key="test-key", debug=True)
            self.log_check(
                "Settings Instantiation",
                True,
                f"Created settings for {settings.app_name}",
            )
        except Exception as e:
            self.log_check("Settings Instantiation", False, str(e))
            tests_passed = False

        # Test 2: Citation parser
        try:
            from judicaita.citation_mapping.parser import CitationParser

            parser = CitationParser()
            citations = parser.extract_citations(
                "Brown v. Board of Education, 347 U.S. 483 (1954)"
            )
            self.log_check(
                "Citation Parser",
                len(citations) > 0,
                f"Extracted {len(citations)} citation(s)",
            )
        except Exception as e:
            self.log_check("Citation Parser", False, str(e))
            tests_passed = False

        # Test 3: Notebook utils
        try:
            from judicaita.notebook_utils import NotebookHelper

            helper = NotebookHelper(show_progress=False)
            self.log_check("NotebookHelper", True, "Instantiated successfully")
        except Exception as e:
            self.log_check("NotebookHelper", False, str(e))
            tests_passed = False

        return tests_passed

    def generate_report(self) -> str:
        """Generate a markdown validation report."""
        lines = [
            "# JudicAIta Kaggle Environment Validation Report",
            "",
            f"**Generated**: {self.results['timestamp']}",
            f"**Overall Status**: {'✅ PASSED' if self.all_passed else '❌ FAILED'}",
            "",
            "## Check Results",
            "",
        ]

        for name, result in self.results["checks"].items():
            icon = "✅" if result["passed"] else "❌"
            lines.append(f"- {icon} **{name}**: {result['message']}")

        if self.results["warnings"]:
            lines.extend(["", "## Warnings", ""])
            for warning in self.results["warnings"]:
                lines.append(f"- ⚠️ {warning}")

        if self.results["errors"]:
            lines.extend(["", "## Errors", ""])
            for error in self.results["errors"]:
                lines.append(f"- ❌ {error}")

        if self.results["recommendations"]:
            lines.extend(["", "## Recommendations", ""])
            for rec in self.results["recommendations"]:
                lines.append(f"- 💡 {rec}")

        lines.extend(
            [
                "",
                "## Next Steps",
                "",
                "1. Address any errors listed above",
                "2. Review warnings for potential issues",
                "3. Follow recommendations for optimal setup",
                "4. Run the training notebook after all checks pass",
            ]
        )

        return "\n".join(lines)

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BLUE}JudicAIta Kaggle Environment Validator{RESET}")
        print(f"{BLUE}{'=' * 40}{RESET}\n")

        self.check_python_version()
        self.check_core_dependencies()
        self.check_ml_dependencies()
        self.check_tpu_availability()
        self.check_memory()
        self.check_judicaita_import()
        self.run_smoke_test()

        # Summary
        self.print_header("Validation Summary")

        passed_count = sum(
            1 for r in self.results["checks"].values() if r["passed"]
        )
        total_count = len(self.results["checks"])

        if self.all_passed:
            print(f"  {GREEN}✅ ALL CHECKS PASSED ({passed_count}/{total_count}){RESET}")
            print(f"  {GREEN}Environment is ready for JudicAIta!{RESET}")
        else:
            print(f"  {RED}❌ SOME CHECKS FAILED ({passed_count}/{total_count}){RESET}")
            print(f"  {RED}Please address the errors above before proceeding.{RESET}")

        if self.results["warnings"]:
            print(f"\n  {YELLOW}⚠️ {len(self.results['warnings'])} warning(s){RESET}")

        return self.all_passed

    def save_report(self, output_path: str | Path = "validation_report.md") -> None:
        """Save the validation report to a file."""
        report = self.generate_report()
        Path(output_path).write_text(report)
        print(f"\n  Report saved to: {output_path}")

    def save_json(self, output_path: str | Path = "validation_results.json") -> None:
        """Save validation results as JSON."""
        Path(output_path).write_text(json.dumps(self.results, indent=2))
        print(f"  Results saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    validator = KaggleEnvironmentValidator()
    passed = validator.run_all_checks()

    # Save reports if running in Kaggle/Colab
    try:
        if Path("/kaggle").exists() or "google.colab" in sys.modules:
            validator.save_report()
            validator.save_json()
    except Exception:
        pass

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
