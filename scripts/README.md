# scripts/

CLI convenience wrappers around the `src/` modules. On POSIX systems
you can also use the top-level `Makefile`.

| Script              | Purpose                                          |
|---------------------|--------------------------------------------------|
| `setup.ps1`         | Create venv + install CPU/base deps (Windows)    |
| `setup.sh`          | Create venv + install CPU/base deps (POSIX)      |
| `prepare_data.ps1`  | Download + convert + validate VisDrone           |

Each script is a thin wrapper; see the file header for the underlying
`python -m ...` commands so you can run steps individually.
