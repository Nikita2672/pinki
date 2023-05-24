def parser_sec_met() -> {}:
    data_dict = {}
    with open('util/data/sec_metro.txt') as file:
        for line in file:
            line = line.strip()
            sector, metro = line.split()

            if sector not in data_dict:
                data_dict[sector] = set()
            data_dict[sector].add(metro)
    return data_dict


def parser_sec_num() -> {}:
    data_dict = {}
    with open('util/data/sector_num.txt') as file:
        for line in file:
            line = line.strip()
            sector, num = line.split()
            data_dict[sector] = int(num)
    return data_dict


def parser_metro_num() -> {}:
    data_dict = {}
    with open('util/data/metro_num.txt') as file:
        for line in file:
            line = line.strip()
            metro, num = line.split()
            data_dict[metro] = int(num)
    return data_dict
