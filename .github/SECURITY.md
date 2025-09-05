# Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **Do not** create a public GitHub issue for security vulnerabilities
2. Email the maintainers directly with details
3. Include steps to reproduce the vulnerability
4. Allow reasonable time for the issue to be addressed before public disclosure

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| develop | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

When contributing to this project:

- Never commit API keys, passwords, or secrets
- Use environment variables for sensitive configuration
- Validate all inputs, especially when processing EEG data
- Follow secure coding practices
- Keep dependencies up to date

## Data Security

This project processes potentially sensitive EEG data:

- Ensure all data is properly anonymized
- Follow institutional data handling policies
- Use secure data transfer methods
- Implement proper access controls

## Vulnerability Response

We commit to:

- Acknowledging receipt of vulnerability reports within 48 hours
- Providing regular updates on remediation progress
- Crediting security researchers (unless they prefer anonymity)
- Releasing security patches promptly
