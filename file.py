import os

def print_structure(base_path=".", max_depth=2, file_name="short_structure.txt"):
    def walk_dir(path, depth=0):
        if depth > max_depth:
            return
        indent = "    " * depth
        f.write(f"{indent}{os.path.basename(path)}/\n")
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    walk_dir(item_path, depth + 1)
                elif depth < max_depth:
                    f.write(f"{'    ' * (depth + 1)}{item}\n")
        except PermissionError:
            pass

    with open(file_name, "w", encoding="utf-8") as f:
        walk_dir(base_path)

print_structure(".", max_depth=2)
print("✅ Folder structure saved to short_structure.txt")
