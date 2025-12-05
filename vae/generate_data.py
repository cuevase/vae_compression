from vae_class import VAE
import os
import glob


def complete_one_cycle_image(path_to_image, path_to_save):
    vae = VAE()
    loaded_image_tensor, image_size = vae.load_image(path_to_image)
    packed, latent_min, latent_max, original_shape = vae.encode_and_pack(loaded_image_tensor)
    vae.unpack_and_decode(packed, original_shape, latent_min, latent_max, image_size, path_to_save)


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all image files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.webp', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    
    processed = 0
    skipped = 0
    
    for i, input_path in enumerate(image_files, 1):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}_output.png")
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"[{i}/{len(image_files)}] Skipping {base_name} (output exists)")
            skipped += 1
            continue
        
        print(f"[{i}/{len(image_files)}] Processing {input_path} -> {output_path}")
        complete_one_cycle_image(input_path, output_path)
        processed += 1
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main(input_folder="inputs", output_folder="outputs")
