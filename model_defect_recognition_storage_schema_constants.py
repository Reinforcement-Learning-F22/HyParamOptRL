class_info_file = "defect_group_info.json"
void_key_path = "фон"
defect_key = "defect"
channel_single = "single"
channel_double = "double"
channel_double_path = "merged"
void_key = "non_defect"

origin_natural = "natural"
origin_artificial = "artificial"
origin_natural_path = "естественн"
origin_artificial_path = "искусственн"

passageway_num_key = -3
channel_name_key = -2
file_name_key = -1
channel_num_key = 1
angle_num_key = 2
position_idx = -1

file_types = {
    defect_key: [defect_key],
    void_key: [void_key],
    "all": [defect_key, void_key]
}
channel_types = {
    channel_single: [channel_single],
    channel_double: [channel_double],
    "all": [channel_single, channel_double]
}

origin_type = {
    origin_natural: [origin_natural],
    origin_artificial: [origin_artificial],
    "all": [origin_natural, origin_artificial]
}
