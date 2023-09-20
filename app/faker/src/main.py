import os
import csv
import argparse
import json

from faker import Faker


# import providers from faker


def create_personal_info(args, header: list, count: int):
    fake = Faker('en_US')
    csv_filepath = "result.csv"
    with open('/data/output/%s' % csv_filepath, 'wt', encoding="UTF-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for _ in range(count):
            csv_writer.writerow([
                fake.name(),
                fake.bothify(text='010-####-####'),
                fake.address(),
                fake.email(),
                fake.job(),
                fake.company(),
                fake.date(),
                # fake.text(),
                # fake.sentence()
            ])
    with open(os.path.join("/data/output", args.output), "wt", encoding="utf-8") as fp:
        json.dump({
            "text_path": csv_filepath
        }, fp)


def create_log_info(args, header: list, count: int):
    fake = Faker('en_US')
    csv_filepath = "result.csv"
    with open('/data/output/%s' % csv_filepath, 'wt', encoding="UTF-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for _ in range(count):
            csv_writer.writerow([
                fake.ipv4(),
                fake.ipv4_private(),
                fake.ascii_company_email(),
                fake.user_name(),
            ])
    with open(os.path.join("/data/output", args.output), "wt", encoding="utf-8") as fp:
        json.dump({
            "text_path": csv_filepath
        }, fp)

def main(args):
    with open("/data/input/%s" % args.input, "rt", encoding="UTF-8") as fp:
        params = json.load(fp)

    count = params['count']
    if params['type'] == 'personal':
        csv_header = ['name', 'phone', 'address', 'email', 'job', 'company', 'date']
        create_personal_info(args, csv_header, count)
    elif params['type'] == 'log':
        csv_header = ['IP', 'Private IP', 'E-mail', 'User Name']
        create_log_info(args, csv_header, count)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
