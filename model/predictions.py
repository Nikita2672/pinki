import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data1 = pd.DataFrame({
    "sector": ['Academicheskiy', 'Academicheskiy', 'Alekseevski', 'Babuskinski', 'Basmanski', 'Basmanski', 'Basmanski',
               'Basmanski', 'Beskudnikovski', 'Beskudnikovski', 'Butirski', 'Butirski', 'Butirski', 'Butirski',
               'Butirski', 'Voikovski', 'Golovinski', 'Danilovski', 'Danilovski', 'Donskoi', 'Donskoi', 'Krasnoselski',
               'Krasnoselski', 'Krasnoselski', 'Krasnoselski', 'Krasnoselski', 'Krasnoselski', 'Meshanski', 'Meshanski',
               'Meshanski', 'Meshanski', 'Mojaiski', 'Nagorni', 'Nagorni', 'Nizhegorodski', 'Obruchevski',
               'Ostankinski', 'Vnukovskoe', 'Moskovski', 'Sosenskoe', 'Sosenskoe', 'Sosenskoe', 'Sosenskoe',
               'Presnenski', 'Presnenski', 'Presnenski', 'Presnenski', 'Presnenski', 'Presnenski', 'Presnenski',
               'Presnenski', 'Arbat', 'Arbat', 'Arbat', 'Arbat', 'Aeroport', 'Aeroport', 'Aeroport', 'Bibirevo',
               'Bibirevo', 'Bogorodskoe', 'Brateevo', 'Vihino-Julebino', 'Vihino-Julebino', 'Vihino-Julebino',
               'Vihino-Julebino', 'Vihino-Julebino', 'Vihino-Julebino', 'Golyanovo', 'Dorogomilovo', 'Dorogomilovo',
               'Dorogomilovo', 'Dorogomilovo', 'Dorogomilovo', 'Dorogomilovo', 'Zamoskvoreche', 'Zamoskvoreche',
               'Zamoskvoreche', 'Zamoskvoreche', 'Zamoskvoreche', 'Zuzino', 'Zuzino', 'Zuzino', 'Zyablikovo',
               'Zyablikovo', 'Zyablikovo', 'Zyablikovo', 'Izmailovo', 'Izmailovo', 'Izmailovo', 'Konkovo',
               'Kosino-Uhtomski', 'Kosino-Uhtomski', 'Krylatskoe', 'Kuzminki', 'Kuzminki', 'Kuncevo', 'Kuncevo',
               'Levoberegni', 'Levoberegni', 'Lefortovo', 'Lefortovo', 'Lublino', 'Marfino', 'Maruina Rocha', 'Maruino',
               'Maruino', 'Mitino', 'Mitino', 'Mitino', 'Moskvoreche-Saburovo', 'Moskvoreche-Saburovo', 'Nekrasovka',
               'Novogireevo', 'Novogireevo', 'Novokosino', 'Novo-Peredelkino', 'Novo-Peredelkino',
               'Orehovo-Borisovo Severnoe', 'Orehovo-Borisovo Uzhnoe', 'Otradnoe', 'Otradnoe', 'Ochakovo-Matveevskoe',
               'Perovo', 'Pechatniki', 'Pokrovskoe-Streshnevo', 'Pokrovskoe-Streshnevo', 'Preobrazhenskoe',
               'Preobrazhenskoe', 'Prospect Vernadskogo', 'Ramenki', 'Ramenki', 'Ramenki', 'Ramenki', 'Ramenki',
               'Sviblovo', 'Sviblovo', 'Severnoe Butovo', 'Severnoe Butovo', 'Severnoe Butovo', 'Severnoe Medvedkovo',
               'Severnoe Tushino', 'Sokol', 'Sokolinaya Gora', 'Sokolinaya Gora', 'Sokolinaya Gora', 'Sokolniki',
               'Solncevo', 'Solncevo', 'Solncevo', 'Strogino', 'Strogino', 'Teplui Stan', 'Troparevo-Nikulino',
               'Troparevo-Nikulino', 'Filevskii Park', 'Filevskii Park', 'Fili-Davidkovo', 'Fili-Davidkovo',
               'Hamovniki', 'Hamovniki', 'Hamovniki', 'Hamovniki', 'Hovrino', 'Horoshevo-Mnevniki', 'Cheremyshki',
               'Certanovo Severnoe', 'Certanovo Centralnoe', 'Certanovo Centralnoe', 'Certanovo Uznoe',
               'Certanovo Uznoe', 'Shukino', 'Shukino', 'Uznoe Butovo', 'Uznoe Butovo', 'Uznoe Butovo', 'Uznoe Butovo',
               'Yakimanka', 'Yakimanka', 'Yasenevo', 'Yasenevo', 'Yasenevo', 'Ryazanski', 'Ryazanski', 'Ryazanski',
               'Savelovski', 'Taganski', 'Taganski', 'Taganski', 'Taganski', 'Taganski', 'Taganski', 'Tverskoi',
               'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi',
               'Tverskoi', 'Tverskoi', 'Tverskoi', 'Tverskoi', 'Timiryazevski', 'Timiryazevski', 'Timiryazevski',
               'Horoshevski', 'Horoshevski', 'Uznoportovi', 'Uznoportovi', 'Uznoportovi', 'Tcaritcyno', 'Tcaritcyno',
               'Tekstilshchiki', 'Nagatino-Sadovniki', 'Nagatino-Sadovniki', 'Uznoe Tushino'],
    "metro": ['Profsoiuznaia', 'Akademicheskaia', 'Alekseevskaia', 'Babushkinskaia', 'Kurskaia', 'Chkalovskaia',
              'Chistye prudy', 'Baumanskaia', 'Seligerskaia', 'Verkhnie Likhobory', 'Fonvizinskaia', 'Dmitrovskaia',
              'Savelovskaia', 'Butyrskaia', 'Timiriazevskaia', 'Voikovskaia', 'Vodnyi stadion', 'Avtozavodskaia',
              'Tulskaia', 'Shabolovskaia', 'Leninskii prospekt', 'Komsomolskaia', 'Turgenevskaia', 'Krasnoselskaia',
              'Sretenskii bulvar', 'Sukharevskaia', 'Krasnye vorota', 'Kuznetckii most', 'Rizhskaia', 'Trubnaia',
              'Prospekt Mira', 'Davydkovo', 'Nagatinskaia', 'Nagornaia', 'Nizhegorodskaia', 'Kaluzhskaia', 'VDNKh',
              'Rasskazovka', 'Rumiantcevo', 'Filatov Lug', 'Prokshino', 'Olkhovaia', 'Kommunarka', 'Mezhdunarodnaia',
              'Begovaia', 'Ulitca 1905 goda', 'Krasnopresnenskaia', 'Vystavochnaia', 'Delovoi tcentr', 'Barrikadnaia',
              'Shelepikha', 'Arbatskaia', 'Smolenskaia', 'Biblioteka im. Lenina', 'Borovitckaia', 'Dinamo', 'Aeroport',
              'Petrovskii park', 'Altufevo', 'Bibirevo', 'Bulvar Rokossovskogo', 'Alma-Atinskaia',
              'Lermontovskii prospekt', 'Kosino', 'Kotelniki', 'Zhulebino', 'Iugo-Vostochnaia', 'Vykhino',
              'Shchelkovskaia', 'Park Pobedy', 'Kievskaia', 'Minskaia', 'Studencheskaia', 'Kievskaia', 'Kutuzovskaia',
              'Paveletckaia', 'Tretiakovskaia', 'Dobryninskaia', 'Serpukhovskaia', 'Novokuznetckaia', 'Kakhovskaia',
              'Sevastopolskaia', 'Nakhimovskii prospekt', 'Krasnogvardeiskaia', 'Borisovo', 'Shipilovskaia',
              'Ziablikovo', 'Pervomaiskaia', 'Partizanskaia', 'Izmailovskaia', 'Beliaevo', 'Lukhmanovskaia',
              'Ulitca Dmitrievskogo', 'Krylatskoe', 'Volzhskaia', 'Kuzminki', 'Kuntcevskaia', 'Molodezhnaia',
              'Rechnoi vokzal', 'Belomorskaia', 'Aviamotornaia', 'Lefortovo', 'Liublino', 'Vladykino', 'Marina Roshcha',
              'Marino', 'Bratislavskaia', 'Volokolamskaia', 'Piatnitckoe shosse', 'Mitino', 'Kashirskaia',
              'Varshavskaia', 'Nekrasovka', 'Novogireevo', 'Perovo', 'Novokosino', 'Novoperedelkino',
              'Borovskoe shosse', 'Orekhovo', 'Domodedovskaia', 'Otradnoe', 'Vladykino', 'Ozernaia',
              'Shosse Entuziastov', 'Pechatniki', 'Spartak', 'Tushinskaia', 'Preobrazhenskaia ploshchad',
              'Cherkizovskaia', 'Prospekt Vernadskogo', 'Lomonosovskii prospekt', 'Michurinskii prospekt',
              'Michurinskii prospekt', 'Universitet', 'Ramenki', 'Botanicheskii sad', 'Sviblovo',
              'Bulvar Dmitriia Donskogo', 'Lesoparkovaia', 'Ulitca Starokachalovskaia', 'Medvedkovo', 'Planernaia',
              'Sokol', 'Elektrozavodskaia', 'Semenovskaia', 'Izmailovskaia', 'Sokolniki', 'Solntcevo', 'Salarevo',
              'Govorovo', 'Miakinino', 'Strogino', 'Konkovo', 'Iugo-Zapadnaia', 'Troparevo', 'Fili', 'Bagrationovskaia',
              'Pionerskaia', 'Slavianskii bulvar', 'Frunzenskaia', 'Park kultury', 'Sportivnaia', 'Kropotkinskaia',
              'Khovrino', 'Narodnoe Opolchenie', 'Novye Cheremushki', 'Chertanovskaia', 'Prazhskaia', 'Iuzhnaia',
              'Ulitca Akademika Iangelia', 'Annino', 'Shchukinskaia', 'Oktiabrskoe pole', 'Bulvar Admirala Ushakova',
              'Ulitca Gorchakova', 'Ulitca Skobelevskaia', 'Buninskaia Alleia', 'Oktiabrskaia', 'Polianka',
              'Teplyi Stan', 'Novoiasenevskaia', 'Iasenevo', 'Okskaia', 'Stakhanovskaia', 'Riazanskii prospekt',
              'Savelovskaia', 'Marksistskaia', 'Ploshchad Ilicha', 'Taganskaia', 'Proletarskaia',
              'Krestianskaia zastava', 'Rimskaia', 'Tverskaia', 'Okhotnyi riad', 'Pushkinskaia', 'Maiakovskaia',
              'Belorusskaia', 'Novoslobodskaia', 'Dostoevskaia', 'Lubianka', 'Teatralnaia', 'Kitai-gorod',
              'Tcvetnoi bulvar', 'Mendeleevskaia', 'Chekhovskaia', 'Timiriazevskaia', 'Okruzhnaia',
              'Petrovsko-Razumovskaia', 'Polezhaevskaia', 'TcSKA', 'Kozhukhovskaia', 'Volgogradskii prospekt',
              'Dubrovka', 'Tcaritcyno', 'Kantemirovskaia', 'Tekstilshchiki', 'Kolomenskaia', 'Tekhnopark',
              'Skhodnenskaia']
})

df = pd.read_csv('last.csv')

labelencoder = LabelEncoder()
df['numsect'] = labelencoder.fit_transform(df['sector'])
df['nummetro'] = labelencoder.fit_transform(df['metro'])

data1['numsect'] = labelencoder.fit_transform(data1['sector'])
data1['nummetro'] = labelencoder.fit_transform(data1['metro'])

data = df

df = df.drop(columns='sector')
df = df.drop(columns='metro')
df = df.drop(columns='fee_percent')
df = df.drop(columns='provider')

x = df.drop(columns='price')
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

tree = DecisionTreeRegressor(random_state=10, max_depth=12, min_samples_leaf=10, max_leaf_nodes=210)
tree.fit(x_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(tree, file)

numsect = 1
nummetro = 2
total_area = 100
rooms = 4
way = 1
minutes = 10
