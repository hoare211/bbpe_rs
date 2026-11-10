# Security policy

## Supported versions

Only the latest release on the default branch receives security fixes.

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately through GitHub's
[Security Advisories](https://github.com/hoare211/bbpe_rs/security/advisories/new)
form, or by email to the maintainers listed in the repository profile.

Include:

- affected version or commit,
- a minimal reproduction (model file, corpus line or command),
- impact assessment (crash, memory exhaustion, wrong output, code execution).

You can expect an acknowledgement within a few business days. Once a fix is
available, the advisory and a patched release will be published.

## Scope

In scope:

- panics or memory unsafety reachable from untrusted model files or corpora,
- excessive memory or CPU use triggered by crafted input,
- path traversal or file overwrite through CLI arguments.

Out of scope:

- denial of service caused by intentionally huge corpora,
- issues in optional dependencies (`torch`, `transformers`) — report those
  upstream.
