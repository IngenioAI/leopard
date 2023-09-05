import csv
import argparse

from faker import Faker

def main(args):
    fake = Faker('ko_KR')

    with open('/data/output/%s' % args.output, 'wt') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['name', 'phone', 'address', 'email', 'job', 'company', 'date'])
        for _ in range(args.count):
            csv_writer.writerow([
                fake.name(),
                fake.bothify(text='010-####-####'),
                fake.address(),
                fake.email(),
                fake.job(),
                fake.company(),
                fake.date(),
                #fake.text(),
                #fake.sentence()
            ])

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default="result.csv")
    return parser.parse_args()            

if __name__ == "__main__":
    main(parse_arguments())