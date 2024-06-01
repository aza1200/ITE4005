import sys
import copy
from collections import Counter
from itertools import permutations

def get_new_ckplusone_list(tmp_lk_dict, target_length):
    ret_list = []

    keys = tmp_lk_dict.keys()
    for a_key in keys:
        for b_key in keys:
            new_key = tuple(sorted(list((set(a_key + b_key)))))
            if len(new_key) == target_length and new_key not in ret_list:
                ret_list.append(new_key)
    return ret_list

# txt 파일 input list 로 변환
def get_input_list(input_file_name):
    input_list = []
    with open(input_file_name, 'r') as file:
        for line in file:
            numbers = list(map(int, line.strip().split()))
            input_list.append(numbers)
    return input_list

def get_lkplusone_dict_from_ckplusone(input_list, ckplusone_list, min_support):
    ret_dict = {}

    for tmp_ck_list in ckplusone_list:
        now_count = 0

        for tmp_input in input_list:
            all_check = True
            for tmp in tmp_ck_list:
                if tmp not in tmp_input: all_check = False
            if all_check: now_count +=1
        
        if now_count >= min_support:
            ret_dict[tmp_ck_list] = now_count
    return ret_dict

# frequent dict 반환인거임 결국
def get_apriori(min_support_percent, input_list):
    ret_dict = {}

    min_support =   (min_support_percent/100)*len(input_list)
    all_itemsets = [element for transaction in input_list for element in transaction]

    c_k_dict = Counter(all_itemsets)
    l_k_dict = {tuple([key]): value for key, value in c_k_dict.items() if value >= min_support}
    ret_dict.update(l_k_dict)

    target_length = 1
    while True:
        target_length +=1

        # l_k_dict 값토대로 c_k+1 만들기 
        ckplustone_list = get_new_ckplusone_list(l_k_dict, target_length)
        
        # c_k+1 토대로 l_k+1_dict 만들기
        lk_plus_one_dict = get_lkplusone_dict_from_ckplusone(input_list, ckplustone_list, min_support)
        l_k_dict = copy.deepcopy(lk_plus_one_dict)
        

        if len(lk_plus_one_dict) == 0: break
        else: ret_dict.update(lk_plus_one_dict)

    return ret_dict

# freqeuent_dict 에서 key쌍 두개 가져오기 리스트 형태로 [[A,B],[A,C]]
def get_permutation_list(frequent_dict):
    keys = list(frequent_dict.keys())
    permutations_list = list(permutations(keys, 2))
    return permutations_list

def check_has_same_element(A, B):
    return bool(set(A).intersection(set(B)))

def count_tuple_count(tmp_tuple, input_list):
    now_count = 0
    for tmp_input in input_list:
        all_check = True
        for tmp in tmp_tuple:
            if tmp not in tmp_input: all_check = False
        if all_check: now_count +=1
    return now_count

def tuple_to_string(tmp_tuple):
    return "{" + ",".join(map(str, tmp_tuple)) + "}"

def get_association_rule(frequent_dict, input_list):
    permutation_pairs = get_permutation_list(frequent_dict)
    
    ret_list = []
    for A,B in permutation_pairs:
        if check_has_same_element(A,B) is True: continue
        all_elements = tuple(sorted(list(A+B)))
        if all_elements not in frequent_dict: continue

        n_count = len(input_list)
        union_count = count_tuple_count(all_elements, input_list)
        a_count = count_tuple_count(A, input_list)

        support = '{:.2f}'.format(round((union_count/n_count)*100,2))
        confidence = '{:.2f}'.format(round((union_count/a_count)*100, 2))

        ret_list.append([A, B, support, confidence])
    return ret_list

def write_to_file(result_list, filename):
    with open(filename, 'w') as file:
        for A,B,support,confidence in result_list:
            A_string = tuple_to_string(A)
            B_string = tuple_to_string(B)
            result_string = f'{A_string}\t{B_string}\t{support}\t{confidence}'
            file.write("%s\n" %result_string)


if __name__ == '__main__':
    min_support_percent = int(sys.argv[1])
    input_file_name = str(sys.argv[2])
    output_file_name = str(sys.argv[3])

    # input file 을 txt 에서 리스트로 읽어옴
    input_list = get_input_list(input_file_name)

    # apriori 알고리즘으로 dictionary 형태로 받아옴 
    frequent_dict = get_apriori(min_support_percent, input_list)
    
    # frequent_dict 기반으로 association rule 가져옴 
    result_list = get_association_rule(frequent_dict, input_list)

    # 결과를 output.txt 파일에 쓰기
    write_to_file(result_list, output_file_name)



