import tarfile
import os

# Path to your Lakh MIDI tar.gz file
tar_path = "../../../../dataset/Raw/Lakh/lmd_matched.tar.gz"

# Output directory where .mid files will be extracted
output_dir = "../../../../dataset/Extracted/Lakh"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open and extract the .mid files from the tar.gz archive
with tarfile.open(tar_path, "r:gz") as tar:
    count = 0
    for member in tar.getmembers():
        if member.name.endswith(".mid"):
            member_path_parts = member.name.split("/")
            filename = "_".join(member_path_parts[1:])  # To keep filenames unique
            output_path = os.path.join(output_dir, filename)
            with tar.extractfile(member) as source, open(output_path, "wb") as target:
                target.write(source.read())
            count += 1
            if count % 1000 == 0:
                print(f"Extracted {count} files")

print("Extraction complete.")