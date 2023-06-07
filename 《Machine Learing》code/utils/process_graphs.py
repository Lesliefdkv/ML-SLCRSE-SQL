import os, json, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.graph_utils import GraphProcessor
import time


def process_dataset_graph(processor, dataset, tables, output_path=None, skip_large=False):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        db = tables[entry['db_id']]
        if skip_large and len(db['column_names']) > 100:
            continue
        if (idx + 1) % 500 == 0:
            print('处理第 %d 个例子 ...' % (idx + 1))
        entry = processor.process_graph_utils(entry, db)
        processed_dataset.append(entry)

    if output_path is not None:
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True)
    arg_parser.add_argument('--table_path', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)
    args = arg_parser.parse_args()

    processor = GraphProcessor()

    tables = pickle.load(open(args.table_path, 'rb'))
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    start_time = time.time()
    dataset = process_dataset_graph(processor, dataset, tables, args.output_path)
    print('数据集预处理costs %.4fs .' % (time.time() - start_time))
