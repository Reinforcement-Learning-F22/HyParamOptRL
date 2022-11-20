import os
import copy
import model_defect_recognition_storage_schema_constants as Storage


class DataPathInfo:
    def __init__(self, file_path, image_channels_count: int = 0, image_height: int = 0, image_width: int = 0):

        # image info
        self.expected_image_shape = (image_channels_count, image_height, image_width)

        # path normalization
        file_path_normed = os.path.normpath(file_path)

        # set normed path
        self.path: str = file_path_normed

        # get components of normed path with OS path separator
        path_components = file_path_normed.split(os.sep)

        # get channel name from normed path components
        self.channel_name: str = path_components[Storage.channel_name_key]

        # split channel name to components
        channel_name_components = self.channel_name.split('_')

        # get attack angle from channel name components
        self.angle: int = int(channel_name_components[Storage.angle_num_key])

        # get channel number from channel name components
        self.channel_num: int = int(channel_name_components[Storage.channel_num_key])

        # get passage way number from path components
        self.passageway_num: str = path_components[Storage.passageway_num_key]

        self.file_type: str = Storage.void_key if Storage.void_key_path in file_path else Storage.defect_key
        self.is_defect_key: bool = False if Storage.void_key_path in file_path else True

        self.channel_type: str = Storage.channel_double if Storage.channel_double_path in file_path else Storage.channel_single

        self.origin_type: str = Storage.origin_natural if Storage.origin_natural_path in file_path else Storage.origin_artificial

        self.file_index: int = int(path_components[Storage.file_name_key].split('_')[Storage.position_idx].split('.')[0])

        self.file_type_path_components = [self.file_type]

        self.channel_type_path_components = copy.deepcopy(self.file_type_path_components)
        self.channel_type_path_components.append(self.channel_type)

        self.origin_type_path_components = copy.deepcopy(self.channel_type_path_components)
        self.origin_type_path_components.append(self.origin_type)

        self.passage_way_path_components = copy.deepcopy(self.origin_type_path_components)
        self.passage_way_path_components.append(self.passageway_num)

        self.channel_name_path_components = copy.deepcopy(self.passage_way_path_components)
        self.channel_name_path_components.append(str(self.channel_num))

    def replace_path_component(self, path_components, new_component, position) -> list:
        new_path_component = copy.deepcopy(path_components)
        new_path_component[position] = new_component
        return new_path_component

