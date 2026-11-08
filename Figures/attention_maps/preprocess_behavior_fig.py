"""
1/6/26
Take figure from behavior paper (Wegrzyn et al 2017), and split it into individual face figures.
Based on code from Transformer-Explainability/example_face_ims/preprocess_nimstim.py.
"""
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

only_crop = False

face_labels = ['m_hap', 'f_hap',
               'm_sad', 'f_sad',
               'm_ang', 'f_ang',
               'm_fea', 'f_fea',
               'm_dis', 'f_dis',
               'm_sup', 'f_sup',
               'm_ntr', 'f_ntr',
               'color_bar']

crop_tops = {'01F_AN_O': 82,
             '01F_DI_O': 82,
             '01F_FE_O': 70,
             '01F_SA_O': 70,
             }


def crop_panel_face(image, box, border_px=3):
    """
    Crop inside a detected panel border, excluding the black boundary.
    """
    x, y, w, h = box

    return image[
        y + border_px : y + h - border_px,
        x + border_px : x + w - border_px
    ]


def detect_panels(image_path, min_area=50_000):
    """
    Detect rectangular face panels in a multi-panel figure.
    Returns panel boxes as (x, y, w, h), sorted left-to-right.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect dark panel borders
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area > min_area and h > w * 0.8:
            boxes.append((x, y, w, h))

    if len(boxes) == 0:
        raise RuntimeError("No panels detected. Try lowering min_area.")

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


def save_split_panels(image_pth, boxes, save_pth):
    image_bgr = cv2.imread(image_pth)
    for i, box in enumerate(boxes):
        face = crop_panel_face(image_bgr, box)
        lbl = face_labels[i]
        if not os.path.isfile(os.path.join(save_pth, f'{lbl}.png')):
            plt.imsave(os.path.join(save_pth, f'{lbl}.png'), cv2.cvtColor(face, cv2.COLOR_BGR2RGB))


def crop_and_resize(input_path: str, save_pth: str, center: bool = False, top: int = 0, size: int = 224) -> None:
    """
    Crop img to a square while keeping the original width.

    Parameters
    ----------
    input_path : str
        Path to input TIFF.
    save_pth : str
        Path to output TIFF.
    center : bool
        Whether to center-crop (if True: ignore 'top).
    top : int
        Starting y-coordinate of the crop.
        The crop will include rows top : top + width.
        Relevant only if center = False
    size: int
        size for resizing image after crop
        (assume it's already square)
    """
    with Image.open(input_path) as img:
        width, height = img.size

        crop_size = width
        top_for_center = (height - crop_size) // 2

        if center:
            print("Center crop.")
            bottom = top_for_center + crop_size
            suf = 'center'
        else:
            bottom = top + crop_size

            if top < 0:
                raise ValueError("top must be >= 0")

            if bottom > height:
                raise ValueError(
                    f"Crop exceeds image height: top + width = {bottom}, "
                    f"but image height is {height}"
                )

            if top > top_for_center:
                print("Crop below center.")
            elif top < top_for_center:
                print("Crop above center.")
            else:
                print("Center crop.")

            suf = f'{top}'

        cropped = img.crop((0, top, width, bottom))
        resized = cropped.resize((size, size))
        resized.save(os.path.join(save_pth, (input_path.split("/")[-1].split(".")[0] + f"_square_{size}_{suf}.png")))


if __name__ == "__main__":
    home_dir = "/home/projects/bagon/ilanaveh"
    if only_crop:
        save_pth = os.path.join(home_dir, 'code/Transformer-Explainability/samples/test_images_nimstim')
        # im_names = ['01F_AN_O', '01F_DI_O', '01F_FE_O']
        im_names = ['01F_SA_O']
        for im in im_names:
            image_pth = os.path.join(home_dir, f"data/NimStim/NimStim/{im}.BMP")
            if isinstance(crop_tops[im], str):
                crop_and_resize(image_pth, save_pth, center=True)
            else:
                crop_and_resize(image_pth, save_pth, top=crop_tops[im])
    else:
        save_pth = 'bhvr_fig_split'
        # image_pth = os.path.join(home_dir, "data/S10Code/mappingTheEmotionalFace-master/figures/mainWeightPlot.png")  # only overlayed maps (from paper's data dir)
        image_pth = os.path.join(home_dir, "code/PaperStuff/Figures/attention_maps/figs_from_bhvr_paper/fig4_weights_with_colorbar.png")  # also with color bar (downloaded from the paper).
        boxes = detect_panels(image_pth)
        save_split_panels(image_pth, boxes, save_pth)
