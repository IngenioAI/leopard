import csv
import argparse
import json

from faker import Faker


def write_csv(args, header: list, count: int, method):
    with open('/data/output/%s' % args.output, 'wt', encoding="UTF-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for _ in range(count):
            csv_writer.writerow(method())


def create_personal_info(args, header: list, count: int):
    fake = Faker('en_US')

    with open('/data/output/%s' % args.output, 'wt', encoding="UTF-8") as csvfile:
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


def create_medical_info(args, header: list, count: int):
    fake = Faker('ko_KR')

    with open('/data/output/%s' % args.output, 'wt', encoding="UTF-8") as csvfile:
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


def main(args):
    with open("/data/input/%s" % args.input, "rt", encoding="UTF-8") as fp:
        params = json.load(fp)

    count = params['count']
    if params['type'] == 'personal':
        csv_header = ['name', 'phone', 'address', 'email', 'job', 'company', 'date']
        create_personal_info(args, csv_header, count)
    elif params['type'] == 'medical':
        csv_header = []
        create_medical_info(args, csv_header, count)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
