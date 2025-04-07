import csv
import datetime
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from database import Train
from database.crud import load_file_by_dataset
from variables import variables


def create_csv(db: Session, load_train: Train, preview = True) -> None:
    datasets = load_train.datasets

    voicemail = []
    human = []
    ring = []
    labels = {}
    countries = {}
    for dataset in datasets:
        countries[dataset.id] = dataset

        files = load_file_by_dataset(db, dataset.id)

        for l in dataset.labels:
            labels[l.id] = l

        for file in files:
            if file.label_id is not None:
                label = labels[file.label_id]

                if label.notation is not None:
                    if label.notation.name == "human":
                        human.append(file)

                    if label.notation.name == "voicemail":
                        voicemail.append(file)

                    if label.notation.name == "ring":
                        ring.append(file)

    epochs = load_train.fold

    Path(os.path.join(variables.file_dir, load_train.name)).mkdir(parents=True, exist_ok=True)

    csv_filename = f"dataset_{datetime.datetime.utcnow().strftime('%Y_%m_%d')}.csv"
    csv_filepath = os.path.join(variables.file_dir, load_train.name, csv_filename)

    voicemail_iteration_count = len(voicemail) // epochs
    human_iteration_count = len(human) // epochs
    ring_iteration_count = len(ring) // epochs

    logging.info(f"==Train: {load_train.name}. \n"
                 f" CSV filename: {csv_filepath}. \n"
                 f"Length of voicemail: {len(voicemail)}. \n"
                 f"Iteration of voicemail: {voicemail_iteration_count} \n"
                 f"Length of human: {len(human)}. \n"
                 f"Iteration of human: {human_iteration_count} \n"
                 f"Length of ring: {len(ring)}. \n"
                 f"Iteration of ring: {ring_iteration_count} \n")

    voicemail_iteration_position = 0
    human_iteration_position = 0
    ring_iteration_position = 0

    headers = ["filename", "fold", "target", "category", "description_id", "description"]

    output_rows = []

    for step in range(1, epochs + 1):
        for file in range(voicemail_iteration_position, voicemail_iteration_count):
            try:
                content = voicemail[voicemail_iteration_position]
                if not preview:
                    source_dir = os.path.join(variables.file_dir, countries[content.dataset_id].user_id, countries[content.dataset_id].country, "voicemail")
                    fold_dir = os.path.join(variables.file_dir, load_train.name, "audio", f"fold{step}")
                    Path(fold_dir).mkdir(parents=True, exist_ok=True)

                    shutil.copyfile(
                        source_dir + "/{}{}".format(content.id, content.extension),
                        fold_dir + "/{}-{}-0{}".format(step, file + 1, content.extension)
                    )
                output_rows.append([
                    "{}-{}-0{}".format(step, file + 1, content.extension),
                    step,
                    0,
                    "voicemail",
                    content.label_id,
                    labels[content.label_id].name,
                ])
            except Exception as e:
                print('Voicemail file exceed reached: %s', str(e))
            voicemail_iteration_position += 1
            voicemail_iteration_count += 1

    for step in range(1, epochs + 1):
        for file in range(human_iteration_position, human_iteration_count):
            try:
                content = human[human_iteration_position]
                if not preview:
                    source_dir = os.path.join(variables.file_dir, countries[content.dataset_id].user_id, countries[content.dataset_id].country, "human")
                    fold_dir = os.path.join(variables.file_dir, load_train.name, "audio", f"fold{step}")
                    Path(fold_dir).mkdir(parents=True, exist_ok=True)

                    shutil.copyfile(
                        source_dir + "/{}{}".format(content.id, content.extension),
                        fold_dir + "/{}-{}-1{}".format(step, file + 1, content.extension)
                    )
                output_rows.append([
                    "{}-{}-1{}".format(step, file + 1, content.extension),
                    step,
                    1,
                    "human",
                    content.label_id,
                    labels[content.label_id].name,
                ])
            except Exception as e:
                print('Human file exceed reached: %s', str(e))

            human_iteration_position += 1
            human_iteration_count += 1

    for step in range(1, epochs + 1):
        for file in range(ring_iteration_position, ring_iteration_count):
            try:
                content = ring[ring_iteration_position]
                if not preview:
                    source_dir = os.path.join(variables.file_dir, countries[content.dataset_id].user_id, countries[content.dataset_id].country, "ring")
                    fold_dir = os.path.join(variables.file_dir, load_train.name, "audio", f"fold{step}")
                    Path(fold_dir).mkdir(parents=True, exist_ok=True)

                    shutil.copyfile(
                        source_dir + "/{}{}".format(content.id, content.extension),
                        fold_dir + "/{}-{}-2{}".format(step, file + 1, content.extension)
                    )
                output_rows.append([
                    "{}-{}-2{}".format(step, file + 1, content.extension),
                    step,
                    2,
                    "ring",
                    content.label_id,
                    labels[content.label_id].name,
                ])
            except Exception as e:
                print('Ring file exceed reached: %s', str(e))

            ring_iteration_position += 1
            ring_iteration_count += 1

    out = pd.DataFrame(output_rows, columns=headers)
    out.to_csv(csv_filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return FileResponse(csv_filepath, media_type='text/csv', filename=csv_filename)
