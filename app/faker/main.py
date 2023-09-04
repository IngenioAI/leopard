from faker import Faker
fake = Faker('ko_KR')

s = fake.name()
print(s)
# 'Lucy Cechtelar'

s = fake.bothify(text='010-####-####')
print(s)
# Phone number

print(fake.email())

print(fake.job())

print(fake.company())

print(fake.date())


s = fake.address()
print(s)
# '426 Jordy Lodge
#  Cartwrightshire, SC 88120-6700'

s = fake.text()
print(s)
# 'Sint velit eveniet. Rerum atque repellat voluptatem quia rerum. Numquam excepturi
#  beatae sint laudantium consequatur. Magni occaecati itaque sint et sit tempore. Nesciunt
#  amet quidem. Iusto deleniti cum autem ad quia aperiam.
#  A consectetur quos aliquam. In iste aliquid et aut similique suscipit. Consequatur qui
#  quaerat iste minus hic expedita. Consequuntur error magni et laboriosam. Aut aspernatur
#  voluptatem sit aliquam. Dolores voluptatum est.
#  Aut molestias et maxime. Fugit autem facilis quos vero. Eius quibusdam possimus est.
#  Ea quaerat et quisquam. Deleniti sunt quam. Adipisci consequatur id in occaecati.
#  Et sint et. Ut ducimus quod nemo ab voluptatum.'
print("-----")
s =fake.sentence()
print(s)