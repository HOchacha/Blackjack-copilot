import argparse
import os
import yaml
import shutil

def mkdir_if_not_exists(path:str)->None:
    if not os.path.exists(path):
        os.mkdir(path)

def write_all_text(path:str, content:str)->None:
    with open(path, "w") as f:
        f.write(content)

def get_class_of_row(row:str)->int:
    tokens = row.split()
    return int(tokens[0])

def in_classes(x:int)->bool:
    return x in classes

def in_classes2(row:str)->bool:
    return in_classes(get_class_of_row(row))

def trim2(lines):
    return [*filter(in_classes2, lines)]

def trim3(filename:str)->str:
    with open(filename) as f:
        lines = f.readlines()
        tlines = trim2(lines)
        if len(tlines) > 0:
            return '\n'.join(tlines)
        return None

def read_yaml() -> dict:
    with open(os.path.join(srcdir, "data.yaml")) as stream:
        return yaml.safe_load(stream)

def write_yaml(data) -> None:
    with open(os.path.join(dstdir, "data.yaml"), "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

def filter_by_list(src, filt):
    ret = []
    for i in filt:
        ret.append(src[i])
    return ret

def get_files(dir:str):
    dirlist = os.listdir(dir)
    ret = [f for f in dirlist if os.path.isfile(os.path.join(dir, f))]
    return ret

if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("-s", required=False, help="source dataset directory",
                        default=r"C:\Blackjack-copilot\ML\yolo\datasets\Playing Cards.v4-fastmodel-resized640-aug3x.yolov8")
    parser.add_argument("-t", required=False, help="target dataset directory",
                        default=r"C:\Blackjack-copilot\ML\yolo\datasets\Playing Cards.0")
    parser.add_argument("-c", required=False, help="classes", default="0")

    args = parser.parse_args()
    classes = [*map(int, args.c.split())]
    srcdir = args.s
    if not os.path.isdir(srcdir):
        raise Exception("Source dataset directory does not exist.")
    dstdir = args.t
    FIELDS=["test", "val", "train"]

    mkdir_if_not_exists(dstdir)
    ym = read_yaml()
    ym["nc"] = len(classes)
    ym["names"] = filter_by_list(ym["names"], classes)
    write_yaml(ym)

    def copy_label_and_image(src_label_dir:str, dst_label_dir:str, src_image_dir:str, dst_image_dir:str) -> None:
        # get_files returns relative paths
        src_labelfiles = get_files(src_label_dir)

        n = len(src_labelfiles)
        current = 0

        for i in src_labelfiles:
            source = os.path.join(src_label_dir, i)
            target = os.path.join(dst_label_dir, i)

            trimmed = trim3(source)

            # If no class remains after filtering
            if trimmed == None:
                pass
            else:
                write_all_text(target, trimmed)

                name = i[:-3]
                image_filename = name + "jpg"
                source = os.path.join(src_image_dir, image_filename)
                target = os.path.join(dst_image_dir, image_filename)
                shutil.copyfile(source, target)
                print("[%d/%d] copied %s"%(current, n, name))

            current+=1

    def do_field(field:str):
        field_dir = ym[field]

        # Assume that field_dir is relative path
        src_ildir = os.path.join(srcdir, field_dir)
        dst_ildir = os.path.join(dstdir, field_dir)
        mkdir_if_not_exists(dst_ildir)

        src_image_dir = os.path.join(src_ildir, "images")
        dst_image_dir = os.path.join(dst_ildir, "images")
        mkdir_if_not_exists(dst_image_dir)

        src_label_dir = os.path.join(src_ildir, "labels")
        dst_label_dir = os.path.join(dst_ildir, "labels")
        mkdir_if_not_exists(dst_label_dir)

        copy_label_and_image(src_label_dir, dst_label_dir, src_image_dir, dst_image_dir)

    for i in FIELDS:
        do_field(i)
