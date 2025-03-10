import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.options.display.width = 0
df = pd.read_excel('bike.xlsx')

def EDA(df):
    print(df)
    print(df.info()) # Число пропущенных значений (= 0)
    for col in df.columns:
        print(df[col].value_counts()) # Смотрим на возможные значения переменных, чтобы понять их смысл
    # Понимаем, какие переменные категориальные, чтобы закодировать их с помощью one-hot

    sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
    plt.title('Тепловая карта')
    plt.show()

    for col in df.columns: # Смотрим графики распределения каждой переменной, ищем выбросы и аномалии
        plt.hist(df[col], bins=30)
        plt.title(col)
        plt.show()
