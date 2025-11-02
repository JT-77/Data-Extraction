from pdf2image import convert_from_path
import os

pdf_dir = "../datasets/KL-NDA/documents"
output_img_dir = "../datasets/KL-NDA/images"
os.makedirs(output_img_dir, exist_ok=True)

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        doc_id = os.path.splitext(pdf_file)[0]

        print(f"Converting {pdf_file}...")
        try:
            doc_output_folder = os.path.join(output_img_dir, doc_id)
            os.makedirs(doc_output_folder, exist_ok=True)

            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                img_filename = f"{doc_id}_page_{i}.jpg"
                image.save(os.path.join(doc_output_folder, img_filename), "JPEG")
            print(f"  Converted {len(images)} pages for {pdf_file}")
        except Exception as e:
            print(f"  Error converting {pdf_file}: {e}")

print("PDF conversion complete.")