# Contributing to EEG Foundation Challenge 2025

We welcome contributions to the EEG Foundation Challenge project! This document provides guidelines for contributing.

## Development Process

1. Fork the repository
2. Create a feature branch from `develop`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Code Standards

### Python
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants

### File Organization
- Keep modules focused and cohesive
- Use clear, descriptive filenames
- Place related functionality together
- Separate concerns properly

## Testing

- Write unit tests for all new functions
- Ensure model tests validate expected behavior
- Test edge cases and error conditions
- Maintain test coverage above 80%

## Documentation

- Update documentation for any API changes
- Include usage examples
- Document configuration options
- Update changelog for significant changes

## Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring

## EEG/ML Specific Guidelines

- Validate all preprocessing steps
- Document model architecture choices
- Include performance benchmarks
- Test with different data configurations
- Ensure reproducibility with random seeds
