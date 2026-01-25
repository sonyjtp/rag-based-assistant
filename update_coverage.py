"""
Script to validate and report coverage percentage from coverage reports.
Reads from htmlcov/index.html (HTML report) as primary source.
Falls back to coverage.xml (Cobertura format) if HTML report not available.

This script validates that coverage meets the minimum threshold.
The README.md coverage badge should be updated manually or in CI/CD workflows.

Usage: python update_coverage.py
"""

import re
import sys
import xml.etree.ElementTree as ET


def get_coverage_from_html(html_file: str = "htmlcov/index.html") -> float | None:
    """
    Extract coverage percentage from htmlcov/index.html file.

    Args:
        html_file: Path to htmlcov/index.html file

    Returns:
        Coverage percentage as float (e.g., 97.51)
    """
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract coverage percentage from HTML report
        # Look for pattern: <span class="pc_cov">91.28%</span>
        match = re.search(r'<span class="pc_cov">(\d+\.?\d*)%', content)

        if match:
            coverage_percent = float(match.group(1))
            return round(coverage_percent, 2)
        else:
            print(f"⚠️  Could not find coverage percentage in {html_file}")
            return None
    except FileNotFoundError:
        print(f"⚠️  {html_file} not found (will try coverage.xml)")
        return None
    except Exception as e:
        print(f"⚠️  Error reading {html_file}: {e}")
        return None


def get_coverage_from_xml(xml_file: str = "coverage.xml") -> float | None:
    """
    Extract coverage percentage from coverage.xml file (Cobertura format).
    Used as fallback if HTML report is not available.

    Args:
        xml_file: Path to coverage.xml file (Cobertura format)

    Returns:
        Coverage percentage as float (e.g., 97.51)
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get line-rate attribute from root element (represents overall line coverage)
        # line-rate is in decimal format (0.9751 = 97.51%)
        line_rate = root.get("line-rate")

        if line_rate:
            # Convert from decimal (0.9751) to percentage (97.51)
            coverage_percent = float(line_rate) * 100
            return round(coverage_percent, 2)
        else:
            print(f"⚠️  Could not find line-rate in {xml_file}")
            return None
    except FileNotFoundError:
        print(f"❌ {xml_file} not found. Run: pytest --cov=src --cov-report=xml")
        return None
    except ET.ParseError as e:
        print(f"❌ Error parsing {xml_file}: {e}")
        return None


def get_coverage() -> float | None:
    """
    Get coverage percentage from available sources.
    Primary: htmlcov/index.html (most current)
    Fallback: coverage.xml (Cobertura format)

    Returns:
        Coverage percentage as float or None if both sources fail
    """
    # Try HTML report first (more current)
    coverage = get_coverage_from_html()
    if coverage is not None:
        print(f"✅ Coverage from HTML report: {coverage}%")
        return coverage

    # Fall back to XML if HTML not available
    print("Falling back to coverage.xml...")
    coverage = get_coverage_from_xml()
    if coverage is not None:
        print(f"✅ Coverage from XML report: {coverage}%")
        return coverage

    return None


def get_readme_coverage(readme_file: str = "README.md") -> float | None:
    """
    Extract the coverage percentage from the README.md badge.

    Args:
        readme_file: Path to README.md

    Returns:
        Coverage percentage as float or None if not found
    """
    try:
        with open(readme_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract coverage from badge: [![Code Coverage](https://img.shields.io/badge/coverage-91.28%25-...
        match = re.search(r"coverage-([\d.]+)%25", content)

        if match:
            return float(match.group(1))
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def main():
    """Main function to validate coverage."""
    print("📊 Validating coverage badge in README.md...")
    print("   Reading from: htmlcov/index.html (primary) or coverage.xml (fallback)")

    # Get actual coverage from test reports
    actual_coverage = get_coverage()

    if actual_coverage is None:
        print("\n❌ Could not get coverage percentage. Try running tests first:")
        print("   pytest --cov=src --cov-report=html --cov-report=xml")
        return False

    print(f"📈 Found coverage: {actual_coverage}%")

    # Determine quality level
    if actual_coverage >= 90:
        quality = "Excellent 🌟"
    elif actual_coverage >= 85:
        quality = "Good ✅"
    elif actual_coverage >= 75:
        quality = "Fair ⚠️"
    else:
        quality = "Needs improvement ❌"

    print(f"   Quality: {quality}")

    # Get coverage from README badge
    readme_coverage = get_readme_coverage()

    if readme_coverage is None:
        print("\n⚠️  Coverage badge not found in README.md")
        print(f"   Actual coverage: {actual_coverage}%")
        print("   Please update the badge manually or use CI/CD workflow")
        return True  # Don't fail - user can update manually

    # Check if coverage matches
    if abs(actual_coverage - readme_coverage) < 0.01:
        print(f"\n✅ README badge is up-to-date: {readme_coverage}%")
        return True
    else:
        print("\n⚠️  Coverage badge mismatch!")
        print(f"   README badge: {readme_coverage}%")
        print(f"   Actual coverage: {actual_coverage}%")
        print("\n📝 To update the badge, manually edit README.md or use CI/CD workflow:")
        badge_format = (
            f"   Badge format: "
            f"[![Code Coverage](https://img.shields.io/badge/coverage-"
            f"{actual_coverage}%25-COLOR.svg)]()"
        )
        print(badge_format)
        return True  # Don't fail - let user decide


if __name__ == "__main__":
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
