from IPython.core.magic import register_line_magic
import subprocess

def load_ipython_extension(ipython):
    @register_line_magic
    def franklin(line):
        packages = line.strip().split()
        if not packages:
            print("Usage: %franklin_add <package-name>")
            return
        result = subprocess.run(["pixi", "add"] + packages, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Added: {', '.join(packages)}")
        else:
            print(f"Error:\n{result.stderr}")