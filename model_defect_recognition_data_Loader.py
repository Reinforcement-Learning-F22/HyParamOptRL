import os
import json
import random
import numpy as np
from tqdm import tqdm
from typing import Callable
import model_defect_recognition_storage_schema_constants as Storage
from model_defect_recognition_data_path_info import DataPathInfo


def operation_name_decorator(operation_name):
    def inner_decorator(func):
        def wrapper(*args):
            print("{:-^80s}".format(" " + operation_name + " start "))
            return func(*args)

        return wrapper

    return inner_decorator


def load_file(file_path, val_type):
    return np.loadtxt(file_path, dtype=val_type, ndmin=2)


def load_files(file_paths: list, val_type):

    files = []

    for file in file_paths:
        files.append(np.loadtxt(file, dtype=val_type, ndmin=2))

    return files


def dump_files_info_json(data: dict, json_save_path: str):
    file_writer = open(json_save_path, "w", encoding="UTF-8")
    file_writer.write(json.dumps(data, indent='\t', ensure_ascii=False, sort_keys=True))
    file_writer.close()


def nested_dict_get(dic: dict, keys: list):
    for key in keys:
        dic = dic[key]
    return dic


def nested_dict_set(dic: dict, action: Callable, keys: list, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    action(dic, keys[-1], value)


def assignment_op(dictionary: dict, key, value): dictionary[key] = value


def increment_op(dictionary: dict, key, value): dictionary[key] += value


def append_op(dictionary: dict, key, value): dictionary[key].append(value)


def nested_keys_exists(element, keys: list):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


class TxtDataSetFilesLoading:
    def __init__(self, dir_name: str, file_ext: str = ".txt", file_decimal_type: type = np.uint8, export_json_path: str = "",
                 override_defect_dir_info=True):
        
        # Definitions
        
        # extension of the file
        self.file_ext = file_ext
        
        # decimal type of the loading file
        self.file_decimal_type = file_decimal_type
        
        # files searching with spicified file extension (default: .txt)
        self.files = self.__find_files(dir_name, file_ext)
        
        # searchhing for storage schema and general defect group info
        self.group_info_files = self.__find_files(dir_name, '.json')

        # self.image_files = self.__find_files(dir_name, (".png", ".jpg"))
        
        # is defect group info was loaded
        self.defect_group_info_flag = False
        

        if not os.path.exists(export_json_path) or override_defect_dir_info:
            self.storage_schema = self.__create_storage_schema()
            dump_files_info_json(self.storage_schema, export_json_path)
            print("Created defect catalog info at: {}".format(export_json_path))
        else:
            if os.path.exists(export_json_path):
                json_file = open(export_json_path, 'r', encoding="UTF-8")
                self.storage_schema = json.load(json_file)
                print("Loaded defect catalog info from: {}".format(export_json_path))
            else:
                raise FileExistsError

        self.loading_strategies = {
            "single_ch": self.load_single_channel,
            "multiple_ch": self.load_multiple_channels
        }

    @operation_name_decorator("loading")
    def load_files(self, files_type_arg, channel_type_arg, loading_strategy_name="single_ch"):

        if len(self.files) == 0 or len(self.storage_schema) == 0:
            raise ValueError("No files to load or storage schema is empty!")

        if loading_strategy_name == "multiple_ch" and files_type_arg != Storage.defect_key:
            raise ValueError("Wrong 'file_type_arg' for selected 'loading_strategy_name'")

        expected_channels_count = 1
        if loading_strategy_name != "single_ch":
            expected_channels_count = nested_dict_get(self.storage_schema, ["image_channels_count"])
        expected_image_width = nested_dict_get(self.storage_schema, ["image_width"])
        expected_image_height = nested_dict_get(self.storage_schema, ["image_height"])
            
        loaded_files = []
        loaded_labels = []
        visited_defect_railways = set()

        loading_strategy_action = nested_dict_get(self.loading_strategies, [loading_strategy_name])
        expected_file_types = nested_dict_get(Storage.file_types, [files_type_arg])
        expected_channel_types = nested_dict_get(Storage.channel_types, [channel_type_arg])

        for file_path in tqdm(self.files):

            path_info = DataPathInfo(file_path, expected_channels_count, expected_image_height, expected_image_width)

            # Checking correctness of file type and channel type
            if path_info.channel_type in expected_channel_types and path_info.file_type in expected_file_types:

                if tuple(path_info.passage_way_path_components) in visited_defect_railways and loading_strategy_name != "single_ch":
                    continue

                visited_defect_railways.add(tuple(path_info.passage_way_path_components))

                err_code, loaded_image, loaded_label = loading_strategy_action(path_info)

                if err_code == 0:

                    loaded_files.extend(loaded_image)

                    if type(loaded_label) is int:
                        loaded_labels.append(loaded_label)
                    else:
                        loaded_labels.extend(loaded_label)

        print("Total files count in dir ({}): {}".format(self.file_ext, self.storage_schema["count"]))
        print("Loaded images files with extension ({}) count: {}".format(self.file_ext, len(loaded_files)))
        print("Labels count: {}".format(len(loaded_labels)))
        print("Skipped files ({}): {}".format(self.file_ext, self.storage_schema["count"] - len(loaded_files)))

        returned_array = np.asarray(loaded_files)

        if loading_strategy_name == "single_ch":
            returned_array = returned_array[:, np.newaxis]

        return returned_array, np.asarray(loaded_labels)

    def load_single_channel(self, path_info: DataPathInfo):

        error_code = 0

        # loading operation
        loaded_array = load_file(path_info.path, self.file_decimal_type)
        loaded_label = nested_dict_get(self.storage_schema, path_info.file_type_path_components + ["label"])

        # Checking correctness of the shape of the loaded file and third axis adding
        try:
            loaded_array = loaded_array.reshape(path_info.expected_image_shape)

        except:
            error_code = -1
            print("Loaded with wrong shape (will be skipped): {}, expected {}, path: {}"
                  .format(loaded_array.shape, path_info.expected_image_shape, path_info.path))
            return error_code, None, None

        return error_code, loaded_array, loaded_label

    def load_multiple_channels(self, path_info: DataPathInfo):

        # DEFINITIONS
        error_code = 0

        scale_factor = 0

        # The maximum images count in single channel
        max_channel_images_count = -1

        expected_channels_count = path_info.expected_image_shape[0]
        # label creation
        label_mask = np.zeros(expected_channels_count, dtype=np.int8)
        # collected channels
        channels = []

        # Defect railway passage definitions
        defect_passage_way = nested_dict_get(self.storage_schema, path_info.passage_way_path_components)
        defect_channels_count = defect_passage_way.get("channels_count")
        defect_label = nested_dict_get(self.storage_schema, path_info.file_type_path_components + ['label'])
        defect_channels_order: list = defect_passage_way.get("channels_order")
        defect_channels_order.sort()

        # ================================== PHASE 1 ================================== #

        # check if channels count in defect railway passage equals to expected.
        # If it is we do not need in same railway passage with void images

        if defect_channels_count == expected_channels_count:

            for iteration, channel_num in enumerate(defect_channels_order):

                channel_num_str = str(channel_num)

                current_channel = defect_passage_way.get(channel_num_str)

                current_channel_count = current_channel.get("count")

                # set label element
                label_mask[iteration] = defect_label

                # finding maximum images count in railway passage
                if current_channel_count > max_channel_images_count:
                    max_channel_images_count = current_channel_count

                channels.append(current_channel)

        else:

            if defect_channels_count > expected_channels_count:
                raise ValueError(
                    "Receive to many channels in defect railway passage than expected, path components in schema: ",
                    path_info.passage_way_path_components)

            if defect_channels_count == 0:
                raise ValueError(
                    "Channels count equals to 0, path components in schema: ",
                    path_info.passage_way_path_components)

            # otherwise we get the same railway passage with void images
            void_passage_way_path_components = path_info.replace_path_component(path_info.passage_way_path_components, Storage.void_key, 0)
            void_file_type_path_components = path_info.replace_path_component(path_info.file_type_path_components, Storage.void_key, 0)
            void_label_path_components = void_file_type_path_components + ['label']

            # if railway passage does not exist with void images, we can not process current defect railway passage
            if not nested_keys_exists(self.storage_schema, void_passage_way_path_components):
                error_code = -1
                return error_code, None, None

            if not nested_keys_exists(self.storage_schema, void_label_path_components):
                error_code = -1
                return error_code, None, None

            void_passage_way = nested_dict_get(self.storage_schema, void_passage_way_path_components)
            void_channels_count = void_passage_way.get("channels_count")
            void_label = nested_dict_get(self.storage_schema, void_label_path_components)

            # if channels count in railway passage with void images does not equal to expected,
            # we can not process current defect railway passage
            if void_channels_count != expected_channels_count:
                error_code = -1
                return error_code, None, None

            # Get channels order and then sort it
            void_channels_order: list = void_passage_way.get("channels_order")
            void_channels_order.sort()

            # iterate over sorted channels
            for iteration, channel_num in enumerate(void_channels_order):

                channel_num_str = str(channel_num)

                current_channel_count = 0

                current_channel = None

                # if corresponding channel number not in defect channels order then
                if channel_num not in defect_channels_order:

                    # get current channel from void passage railway
                    current_channel = void_passage_way.get(channel_num_str)

                    # images count in current channel
                    current_channel_count = current_channel.get("count")

                    # set label
                    label_mask[iteration] = void_label

                else:

                    # get current channel from defect passage railway
                    current_channel = defect_passage_way.get(channel_num_str)

                    # images count in current channel
                    current_channel_count = current_channel.get("count")

                    # set label element
                    label_mask[iteration] = defect_label

                channels.append(current_channel)

                # finding maximum images count in railway passage
                if current_channel_count > max_channel_images_count:
                    max_channel_images_count = current_channel_count

        images_count = (scale_factor + 1) * max_channel_images_count
        images_shape = tuple([images_count]) + path_info.expected_image_shape

        loaded_files = np.zeros(images_shape, dtype=np.float32)
        loaded_labels = np.full((images_count, expected_channels_count), label_mask, dtype=np.int8)

        # ================================== PHASE 2 ================================== #

        for iteration, channel in enumerate(channels):

            channel_files: list = channel.get("files")

            channel_images_count: int = channel.get("count")

            num_indexes_to_generate = max_channel_images_count - channel_images_count + max_channel_images_count * scale_factor

            random_choice_files = []

            if num_indexes_to_generate > 0:
                random_choice_files = random.choices(channel_files, k=num_indexes_to_generate)
                channel_files.extend(random_choice_files)

            random.shuffle(channel_files)

            # load files
            loaded_files[:, iteration, :, :] = load_files(channel_files, self.file_decimal_type)

        # print(path_info.path)
        # print("label: ", label_mask)
        # print("max count: ", max_channel_images_count)
        # print("images_shape: ", images_shape)

        return error_code, loaded_files, loaded_labels

    def __create_storage_schema(self) -> dict:

        # create storage schema
        storage_schema = dict()

        # set images counter
        nested_dict_set(storage_schema, assignment_op, keys=["count"], value=0)

        for info_file in self.group_info_files:

            # check that current file is defect group info file
            if Storage.class_info_file in info_file:

                # check that defect group file already exists
                if self.defect_group_info_flag:
                    print("Defect group info duplicate was detected. Current file will override existing file")

                # open defect group file
                json_file = open(info_file)

                # load file as json
                json_loaded_file = json.load(json_file)

                # set to the storage schema images info from defect group file
                nested_dict_set(storage_schema, assignment_op, keys=["image_height"],
                           value=json_loaded_file["image_height"])
                nested_dict_set(storage_schema, assignment_op, keys=["image_width"],
                           value=json_loaded_file["image_width"])
                nested_dict_set(storage_schema, assignment_op, keys=["image_channels_count"],
                           value=json_loaded_file["image_channels_count"])

                # get class labels
                self.defect_label = json_loaded_file.get("defect_label")
                self.void_label = json_loaded_file.get("non_defect_label")

                # set flag that defect group file was loaded
                self.defect_group_info_flag = True

        if not self.defect_group_info_flag:
            print("ERROR: Can not find {} file name".format(Storage.class_info_file))
            return {}

        # current file index
        file_path_position = 0

        # iteration over all existing files
        while file_path_position != (len(self.files)):

            # get current file path
            file_path = self.files[file_path_position]

            # check file extension. Default extension .txt
            if not file_path.endswith(self.file_ext):

                print("Unexpected file (will be remove from the list): {}".format(file_path))

                # Remove file from files list
                self.files.remove(file_path)

            else:

                # get all necessary info about current path of the file
                path_info = DataPathInfo(file_path)

                # get valid label value
                label = self.void_label if Storage.void_key_path in file_path else self.defect_label

                # set default values
                if not nested_keys_exists(storage_schema, path_info.file_type_path_components):
                    nested_dict_set(storage_schema, assignment_op, path_info.file_type_path_components, {"label": label})

                if not nested_keys_exists(storage_schema, path_info.passage_way_path_components):
                    nested_dict_set(storage_schema, assignment_op, path_info.passage_way_path_components,
                               {"channels_order": [], "channels_count": 0})

                if not nested_keys_exists(storage_schema, path_info.channel_name_path_components):
                    nested_dict_set(storage_schema, assignment_op, path_info.channel_name_path_components,
                               {"count": 0, "files": []})

                # increment total count
                nested_dict_set(storage_schema, increment_op, ["count"], 1)

                # increment images count in channel
                nested_dict_set(storage_schema, increment_op, path_info.channel_name_path_components + ["count"], 1)

                # set channel's angle
                nested_dict_set(storage_schema, assignment_op, path_info.channel_name_path_components + ["angle"],
                           path_info.angle)

                # set channel's number
                nested_dict_set(storage_schema, assignment_op, path_info.channel_name_path_components + ["channel_number"],
                           path_info.channel_num)

                # add channel's number  
                nested_dict_set(storage_schema, append_op, path_info.channel_name_path_components + ["files"],
                           path_info.path)

                # check if channel number not in channel_order list
                if path_info.channel_num not in nested_dict_get(storage_schema,
                                                           path_info.passage_way_path_components + ["channels_order"]):
                    # append channel number to channel_order list
                    nested_dict_set(storage_schema, append_op, path_info.passage_way_path_components + ["channels_order"],
                               path_info.channel_num)

                    # increment images count in channel
                    nested_dict_set(storage_schema, increment_op, path_info.passage_way_path_components + ["channels_count"],
                               1)

                file_path_position += 1

        return storage_schema

    def __find_files(self, catalog, file_extension):
        founded_files = []
        for root, dirs, files in os.walk(catalog):
            files = [os.path.join(root, name) for name in files if name.endswith(file_extension)]
            founded_files.extend(files)
        print("Found {} files ({}) in {}".format(file_extension, len(founded_files), catalog))
        return founded_files
