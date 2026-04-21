import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def parse_xml_to_yolo(xml_path: Path, class_id: int = 0):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    logging.basicConfig(filename="unmatched_xml.log", level=logging.WARNING)

    xml_dir = Path("data/xml")
    image_dir = Path("data/images/Phone")

    xml_stems = {p.stem: p for p in xml_dir.glob("*.xml")}
    image_stems = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in image_dir.glob(ext):
            image_stems[p.stem] = p

    matched = set(xml_stems) & set(image_stems)
    unmatched = set(xml_stems) - set(image_stems)

    if unmatched:
        logging.warning(f"{len(unmatched)} XML files have no matching image: {sorted(unmatched)}")

    print(f"Matched pairs: {len(matched)}  |  Unmatched XMLs: {len(unmatched)}")

    if not matched:
        print("No matched XML/image pairs found. Expect XMLs under data/xml and images under data/images/Phone.")
        return

    stems = sorted(matched)
    train_val, test_stems = train_test_split(stems, test_size=0.10, random_state=42)
    train_stems, val_stems = train_test_split(train_val, test_size=0.111, random_state=42)
    splits = {"train": train_stems, "val": val_stems, "test": test_stems}

    yolo_root = Path("data/phone_yolo")
    for split, stems_list in splits.items():
        (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for stem in stems_list:
            # Copy image
            shutil.copy(
                image_stems[stem],
                yolo_root / split / "images" / image_stems[stem].name,
            )
            # Write label file
            yolo_lines = parse_xml_to_yolo(xml_stems[stem], class_id=0)
            (yolo_root / split / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines))

    data_yaml = {
        "path": str(yolo_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["phone"],
    }
    with open(yolo_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("data.yaml written. Split sizes:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()

