# UV Commands Reference Guide

A comprehensive guide to UV (Universal Virtualenv) - the fast and reliable Python package manager and environment tool.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Project Initialization](#project-initialization)
4. [Basic Package Management](#basic-package-management)
5. [Environment Management](#environment-management)
6. [Dependency Resolution](#dependency-resolution)
7. [Lock Files and Reproducibility](#lock-files-and-reproducibility)
8. [Python Version Management](#python-version-management)
9. [Advanced Package Management](#advanced-package-management)
10. [Production Deployment](#production-deployment)
11. [CI/CD Integration](#cicd-integration)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)
14. [Quick Reference](#quick-reference)

---

## Introduction

UV is a fast, modern Python package manager and virtual environment tool designed to replace pip, virtualenv, and other traditional Python environment management tools. It provides:

- **Lightning-fast dependency resolution** using a SAT solver
- **Unified tool** for environments, dependencies, and project management
- **Advanced dependency resolution** with conflict detection
- **Lock file support** for reproducible builds
- **Cross-platform compatibility**

---

## Installation and Setup

### Installation Methods

#### Using pipx (Recommended)
```bash
# Install UV using pipx
pipx install uv

# Verify installation
uv --version
```

#### Using pip
```bash
# Install UV using pip
pip install uv

# Verify installation
uv --version
```

#### Using the official installer
```bash
# Download and install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# For Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### Initial Configuration

```bash
# Set up UV cache directory (optional)
export UV_CACHE_DIR=/path/to/cache

# Enable verbose output for debugging
export UV_VERBOSE=1

# Configure Python mirror (optional)
export UV_INDEX_URL=https://pypi.org/simple
export UV_EXTRA_INDEX_URL=https://pypi.python.org/simple
```

---

## Project Initialization

### Initialize a New Python Project

```bash
# Create a new project with all defaults
uv init

# Initialize with specific Python version
uv init --python 3.11

# Initialize in a specific directory
uv init my-project

# Initialize with requirements file
uv init --requirements requirements.txt

# Initialize with dev requirements
uv init --dev
```

### Initialize from Existing Files

```bash
# Initialize from requirements.txt
uv init --requirements requirements.txt

# Initialize from pyproject.toml
uv init --pyproject

# Initialize from setup.py
uv init --setup-py

# Initialize with requirements-dev.txt for development
uv init --requirements requirements.txt --dev-requirements requirements-dev.txt
```

### Project Structure

UV creates the following structure:
```
project/
├── pyproject.toml          # Project configuration
├── .python-version         # Python version specification
├── src/                    # Source code directory
│   └── project_name/       # Package directory
└── tests/                  # Test directory (optional)
```

---

## Basic Package Management

### Installing Packages

#### From PyPI
```bash
# Install a single package
uv add requests

# Install multiple packages
uv add requests pandas numpy

# Install specific version
uv add "requests>=2.25.0"
uv add "requests==2.28.0"

# Install with version constraints
uv add "django<4.0,>=3.2"
uv add "numpy~=1.24.0"  # Compatible release
```

#### From Local Sources
```bash
# Install from local directory
uv add ./local-package

# Install from git repository
uv add git+https://github.com/user/repo.git
uv add git+https://github.com/user/repo.git@branch-name
uv add git+https://github.com/user/repo.git@v1.0.0

# Install editable package
uv add -e ./local-package
```

#### From Private Sources
```bash
# Install with authentication
uv add --index-url https://private.pypi.com/simple/ package-name

# Install with token authentication
export UV_INDEX_URL=https://token@private.pypi.com/simple/
uv add package-name

# Install with username/password
export UV_INDEX_URL=https://username:password@private.pypi.com/simple/
uv add package-name
```

### Installing Development Dependencies

```bash
# Install regular dependencies
uv add pytest black flake8

# Install development dependencies (separate file)
uv add --dev pytest black flake8 mypy

# Install test dependencies only
uv add --test pytest pytest-cov

# Install documentation dependencies
uv add --docs sphinx sphinx-rtd-theme
```

### Installing from Multiple Sources

```bash
# Install from requirements.txt
uv add -r requirements.txt

# Install from multiple requirements files
uv add -r requirements.txt -r requirements-dev.txt

# Install from pyproject.toml
uv add --from pyproject.toml

# Install from setup.py
uv add --from setup.py
```

---

## Environment Management

### Creating Virtual Environments

```bash
# Create virtual environment in default .venv directory
uv venv

# Create with specific Python version
uv venv --python 3.11
uv venv --python 3.10

# Create in specific location
uv venv my-env

# Create with system site packages
uv venv --system-site-packages

# Create with specific pip version
uv venv --pip 23.0.1
```

### Activating Environments

```bash
# Activate environment (Unix/Linux/macOS)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate

# Use UV to run commands without activation
uv run python script.py
uv run pytest
uv run black .
```

### Environment Information

```bash
# Show environment information
uv info

# Show detailed environment info
uv info --verbose

# Show Python version
uv python

# Show installed packages
uv pip list

# Show outdated packages
uv pip list --outdated
```

### Removing Environments

```bash
# Remove current environment
uv venv --rm

# Remove environment in specific directory
uv venv --rm path/to/venv
```

---

## Dependency Resolution

### Understanding Dependency Resolution

UV uses a SAT (Satisfiability) solver for dependency resolution, providing:

- **Faster resolution** compared to pip's legacy resolver
- **Better conflict detection** with clear error messages
- **Deterministic builds** with lock files

### Basic Resolution Commands

```bash
# Resolve dependencies without installing
uv lock --no-install

# Resolve and install dependencies
uv lock

# Update specific packages
uv lock --upgrade requests

# Update all packages
uv lock --upgrade

# Update with specific constraints
uv lock --upgrade --constraint requirements.txt
```

### Advanced Resolution Options

```bash
# Resolution with specific strategy
uv lock --resolution lowest
uv lock --resolution highest
uv lock --resolution compatible

# Resolution with prerelease versions
uv lock --prerelease allow
uv lock --prerelease force

# Resolution with specific Python version
uv lock --python 3.11

# Resolution with sources
uv lock --index-url https://pypi.org/simple
```

### Dependency Tree Analysis

```bash
# Show dependency tree
uv tree

# Show dependency tree with specific package
uv tree requests

# Show dependency tree with conflicts
uv tree --conflicts

# Show dependency tree in JSON format
uv tree --json
```

### Conflict Resolution

```bash
# Detect dependency conflicts
uv tree --conflicts

# Show why package is included
uv tree --why pandas

# Show packages that require specific version
uv tree --why-not requests==2.20.0
```

---

## Lock Files and Reproducibility

### Lock File Management

```bash
# Generate lock file
uv lock

# Update lock file
uv lock --upgrade

# Update lock file with specific package
uv lock --upgrade requests

# Use existing lock file for installation
uv sync --frozen

# Create new lock file
uv lock --generate
```

### Lock File Options

```bash
# Generate minimal lock file
uv lock --generate-hashes

# Lock with specific Python version
uv lock --python 3.11

# Lock with dev dependencies
uv lock --dev

# Lock without dev dependencies
uv lock --no-dev

# Lock from existing environment
uv lock --from-environment .venv
```

### Reproducible Installations

```bash
# Install from lock file (guaranteed reproducible)
uv sync --frozen

# Install with verification
uv sync --frozen --check-hashes

# Install from lock file with extras
uv sync --frozen --all-extras

# Install development dependencies
uv sync --frozen --dev

# Install from specific environment
uv sync --frozen --from-environment .venv
```

### Lock File Maintenance

```bash
# Update lock file without changing packages
uv lock --refresh

# Remove unused dependencies from lock file
uv lock --prune

# Validate lock file consistency
uv lock --check

# Show lock file differences
uv lock --diff
```

---

## Python Version Management

### Python Version Discovery

```bash
# List available Python versions
uv python list

# Find specific Python version
uv python find 3.11

# Show current Python version
uv python

# Show Python executable path
uv python --show-path
```

### Python Version Installation

```bash
# Install specific Python version
uv python install 3.11

# Install with system-specific name
uv python install system

# Install latest patch version
uv python install 3.11

# Install all available versions
uv python install all
```

### Project Python Version

```bash
# Set project Python version
uv python set 3.11

# Set from .python-version file
uv python set --from-file

# Set from current environment
uv python set --from-environment

# Remove Python version specification
uv python unset
```

### Python Version Requirements

```bash
# Require specific Python version in pyproject.toml
uv python require 3.9

# Require range of Python versions
uv python require ">=3.8,<3.12"

# Show Python version requirements
uv python require
```

---

## Advanced Package Management

### Installing with Constraints

```bash
# Install with local constraints file
uv add requests --constraints constraints.txt

# Install with inline constraints
uv add "requests<3.0" --constraint pandas>=1.5

# Install with URL constraints
uv add requests --constraint-url https://example.com/constraints.txt
```

### Extras and Optional Dependencies

```bash
# Install package with extras
uv add "requests[security,socks]"

# Install development extras
uv add --dev "package[dev,test,docs]"

# Install all extras
uv add --all-extras package

# Show available extras
uv show package
```

### Conditional Dependencies

```bash
# Platform-specific dependencies
uv add "platform: windows" pywin32
uv add "platform: linux" systemd-python

# Python version dependencies
uv add "python_version >= '3.11'" typing-extensions
uv add "python_version < '3.9'" importlib-metadata
```

### Private Package Installation

```bash
# Install from private PyPI
uv add --index-url https://private-pypi.com/simple/ private-package

# Install with authentication
export UV_INDEX_URL=https://username:token@private-pypi.com/simple/
uv add private-package

# Install from private git
uv add git+https://github.com/private/repo.git
```

### Custom Package Sources

```bash
# Use multiple index URLs
uv add --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ package

# Install with trusted host
uv add --trusted-host pypi.org --trusted-host files.pythonhosted.org package

# Install from local directory
uv add --find-links ./local-packages package
```

---

## Production Deployment

### Production Environment Setup

```bash
# Create production environment
uv venv --python 3.11 prod-env

# Install from lock file (production)
uv sync --frozen --from-environment prod-env

# Install only production dependencies (no dev)
uv sync --frozen --no-dev --from-environment prod-env

# Install with security checks
uv sync --frozen --check-hashes --from-environment prod-env
```

### Docker Integration

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /usr/local/bin/uv /usr/local/bin/

# Copy project files
COPY pyproject.toml uv.lock ./

# Create and sync environment
RUN uv venv --python 3.11
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "app.py"]
```

### Deployment Scripts

```bash
#!/bin/bash
# deploy.sh

# Set production environment
export ENVIRONMENT=production

# Install production dependencies
uv sync --frozen --no-dev

# Run database migrations
uv run alembic upgrade head

# Collect static files (Django)
uv run python manage.py collectstatic --noinput

# Start application
uv run gunicorn app:app
```

### Production Monitoring

```bash
# Install monitoring dependencies
uv add --prod prometheus-client sentry-sdk

# Install security scanning tools
uv add --dev safety bandit semgrep

# Run security checks
uv run safety check
uv run bandit -r .
uv run semgrep --config=auto .
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v4

    - name: Install UV
      uses: astral-sh/setup-uv@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Setup environment
      run: uv venv --python ${{ matrix.python-version }}

    - name: Install dependencies
      run: uv sync --frozen --dev

    - name: Run tests
      run: uv run pytest
```

### GitLab CI

```yaml
# .gitlab-ci.yml
image: python:3.11

before_script:
  - pip install uv

test:
  script:
    - uv venv --python 3.11
    - uv sync --frozen --dev
    - uv run pytest
    - uv run black --check .
    - uv run flake8 .
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'pip install uv'
                sh 'uv venv --python 3.11'
            }
        }

        stage('Install') {
            steps {
                sh 'uv sync --frozen --dev'
            }
        }

        stage('Test') {
            steps {
                sh 'uv run pytest'
                sh 'uv run black --check .'
                sh 'uv run flake8 .'
            }
        }
    }
}
```

### Pre-commit Integration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/uv-pre-commit
    rev: v0.1.0
    hooks:
      - id: uv-sync
        args: [--frozen]
      - id: uv-run
        args: [black, --check, .]
      - id: uv-run
        args: [flake8, .]
      - id: uv-run
        args: [pytest]
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Dependency Conflicts

```bash
# Problem: Conflicting package versions
uv tree --conflicts

# Solution: Update conflicting packages
uv lock --upgrade

# Solution: Add constraints
uv add "package<3.0" --constraint conflicting-package>=2.0
```

#### Issue: Slow Resolution

```bash
# Problem: Slow dependency resolution
# Solution: Use local cache
export UV_CACHE_DIR=~/.cache/uv

# Solution: Pre-resolve dependencies
uv lock --no-install

# Solution: Use parallel resolution
export UV_RESOLUTION_PARALLEL=true
```

#### Issue: Network Issues

```bash
# Problem: Network timeouts
# Solution: Use mirror
export UV_INDEX_URL=https://pypi.python.org/simple/

# Solution: Increase timeout
export UV_TIMEOUT=300

# Solution: Use retry with backoff
uv add package --retry 3
```

#### Issue: Python Version Mismatch

```bash
# Problem: Python version not found
uv python list

# Problem: Wrong Python version in project
uv python set 3.11

# Problem: Check Python version requirements
uv python require
```

#### Issue: Permission Errors

```bash
# Problem: Permission denied on cache
# Solution: Use user cache directory
export UV_CACHE_DIR=~/.local/share/uv

# Problem: Permission denied on venv
# Solution: Specify user directory
uv venv --python 3.11 --home-dir ~/.local/share/venv
```

### Debug Commands

```bash
# Verbose output
uv --verbose add requests

# Show resolution process
uv add requests --resolution verbose

# Show dependency resolution steps
UV_LOG=debug uv add requests

# Show cached operations
uv cache list

# Clear cache
uv cache clear

# Validate environment
uv validate
```

### Performance Optimization

```bash
# Enable parallel resolution
export UV_RESOLUTION_PARALLEL=true

# Use faster resolver
uv lock --resolution fastest

# Pre-compile wheels
uv add --compile package

# Use binary packages when available
uv add --prefer-binary package

# Disable dev dependencies for speed
uv sync --frozen --no-dev
```

---

## Best Practices

### Project Structure

```
project/
├── pyproject.toml          # Project configuration
├── uv.lock                # Lock file (commit to version control)
├── .python-version        # Python version (commit to version control)
├── .gitignore            # Ignore virtual environments and cache
├── README.md             # Project documentation
├── CHANGELOG.md          # Version history
├── src/                  # Source code
│   └── project_name/     # Package
├── tests/                # Test files
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

### Dependency Management

```toml
# pyproject.toml example
[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "requests>=2.25.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
]
```

### Version Control

```bash
# Always commit these files
git add pyproject.toml uv.lock .python-version

# Always ignore these directories
echo ".venv/" >> .gitignore
echo "uv.lock.backup" >> .gitignore
echo ".uv/" >> .gitignore
```

### Security Best Practices

```bash
# Regularly audit dependencies
uv run safety check

# Use hash verification
uv sync --check-hashes

# Pin critical dependencies
uv add "django==4.2.0" --security

# Regular security updates
uv lock --upgrade
uv run safety check
```

### Performance Best Practices

```bash
# Use lock files in production
uv sync --frozen

# Minimize dev dependencies in production
uv sync --frozen --no-dev

# Use binary packages when possible
uv add --prefer-binary package

# Compile Python files
uv add --compile package
```

### Development Workflow

```bash
# 1. Clone repository
git clone https://github.com/user/project.git
cd project

# 2. Create development environment
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv sync --dev

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Start development
uv run python script.py
```

---

## Quick Reference

### Essential Commands

| Command | Description | Example |
|---------|-------------|---------|
| `uv init` | Initialize new project | `uv init my-project` |
| `uv add` | Install package | `uv add requests` |
| `uv sync` | Install from lock file | `uv sync --frozen` |
| `uv venv` | Create virtual environment | `uv venv` |
| `uv run` | Run command in environment | `uv run pytest` |
| `uv lock` | Generate/update lock file | `uv lock` |
| `uv tree` | Show dependency tree | `uv tree --conflicts` |

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dev` | Development dependencies | `uv add --dev pytest` |
| `--frozen` | Use exact versions from lock | `uv sync --frozen` |
| `--python` | Specify Python version | `uv venv --python 3.11` |
| `--index-url` | Custom package index | `uv add --index-url https://private.pypi.com/simple/` |
| `--constraint` | Add version constraint | `uv add requests --constraint django<4.0` |
| `--upgrade` | Upgrade packages | `uv lock --upgrade` |

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `UV_CACHE_DIR` | Cache directory | `export UV_CACHE_DIR=~/.cache/uv` |
| `UV_INDEX_URL` | Package index URL | `export UV_INDEX_URL=https://private.pypi.com/simple/` |
| `UV_TIMEOUT` | Network timeout | `export UV_TIMEOUT=300` |
| `UV_VERBOSE` | Verbose output | `export UV_VERBOSE=1` |

### File Locations

| File | Purpose | Location |
|------|---------|----------|
| `pyproject.toml` | Project configuration | Project root |
| `uv.lock` | Dependency lock file | Project root |
| `.python-version` | Python version specification | Project root |
| `.venv/` | Virtual environment | Project root (default) |
| `uv.lock.backup` | Lock file backup | Project root |

---

## Conclusion

UV provides a modern, fast, and reliable alternative to traditional Python package management tools. This reference guide covers the essential commands and patterns for effective Python project management with UV.

For the most up-to-date information and additional features, consult:
- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
- [UV Discord Community](https://discord.gg/uv)

Remember to:
- Always use lock files for reproducibility
- Keep your Python version specification in version control
- Use separate environments for development and production
- Regularly audit your dependencies for security issues
- Leverage UV's fast resolution for better development workflows

This guide serves as a comprehensive reference for developers working with UV in any project, from simple scripts to complex enterprise applications.
