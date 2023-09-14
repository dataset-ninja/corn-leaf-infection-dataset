# https://www.kaggle.com/datasets/qramkrishna/corn-leaf-infection-dataset

import csv
import os
from collections import defaultdict

import cv2
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
    mkdir,
    remove_dir,
)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "Corn Leaf Infection"
    dataset_path = "/mnt/d/datasetninja-raw/corn-leaf-infection/Corn Disease detection"
    anns_path = "/mnt/d/datasetninja-raw/corn-leaf-infection/Annotation-export.csv"
    batch_size = 5
    ds_name = "ds"
    images_ext = ".jpg"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        tag_meta = subfolder_to_meta[subfolder]
        tag = sly.Tag(tag_meta)

        if subfolder == "Infected":
            bboxes = im_name_to_bboxes[get_file_name_with_ext(image_path)]

            for bbox in bboxes:
                left = int(bbox[0])
                top = int(bbox[1])
                right = int(bbox[2])
                bottom = int(bbox[3])
                rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
                label = sly.Label(rect, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])

    obj_class = sly.ObjClass("infected leaf", sly.Rectangle)

    tag_healthy = sly.TagMeta("healthy", sly.TagValueType.NONE)
    tag_infected = sly.TagMeta("infected", sly.TagValueType.NONE)

    subfolder_to_meta = {"Healthy corn": tag_healthy, "Infected": tag_infected}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[tag_healthy, tag_infected])
    api.project.update_meta(project.id, meta.to_json())

    im_name_to_bboxes = defaultdict(list)

    with open(anns_path, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue

            im_name_to_bboxes[row[0]].append(list(map(float, row[1:-1])))

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    for subfolder in os.listdir(dataset_path):
        if subfolder == "temp":
            continue
        curr_images_path = os.path.join(dataset_path, subfolder)

        images_names = [
            im_name
            for im_name in os.listdir(curr_images_path)
            if get_file_ext(im_name) == images_ext
        ]

        progress = sly.Progress(
            "Create dataset {}, add {} data".format(ds_name, subfolder), len(images_names)
        )

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(curr_images_path, image_name) for image_name in images_names_batch
            ]

            # TODO =========================== must have, check EXIF Rotate 90 =========================
            temp_img_pathes_batch = []
            temp_folder = os.path.join(dataset_path, "temp")
            mkdir(temp_folder)
            for im_path in img_pathes_batch:
                temp_img = cv2.imread(
                    im_path,
                    flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION,
                )
                new_img_path = os.path.join(temp_folder, get_file_name_with_ext(im_path))
                temp_img_pathes_batch.append(new_img_path)
                cv2.imwrite(new_img_path, temp_img)

            # TODO =======================================================================================

            if subfolder == "Infected":
                images_names_batch = ["infected" + "_" + im_name for im_name in images_names_batch]
            img_infos = api.image.upload_paths(
                dataset.id, images_names_batch, temp_img_pathes_batch
            )
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in temp_img_pathes_batch]
            remove_dir(temp_folder)
            api.annotation.upload_anns(img_ids, anns)
            progress.iters_done_report(len(images_names_batch))
    return project
