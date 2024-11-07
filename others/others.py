import pandas as pd
import numpy as np

# series = processed_model_recommendation_df[score_col].copy()
# series = stats.zscore(series, nan_policy='raise')
# series = stats.norm.cdf(series)
# processed_model_recommendation_df[score_col] = series



def create_bin(series, bins) -> pd.Series:
    bin_labels = list()
    for i, nbin in enumerate(bins):
        prefix = str(i).rjust(3, '0')
        if i == 0:
            bin_labels.append(f'{prefix} <{nbin}')
        elif i + 1 == len(bins):
            bin_labels.append(f'{prefix} >{nbin}')
        else:
            bin_labels.append(f'{prefix} {bins[i-1]+1}-{bins[i]}')

    return pd.cut(
        series,
        bins=[-np.inf] + bins[:-1] + [np.inf],
        labels=bin_labels,
    )


def make_ordinal(n) -> str:
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def check_thai_national_id(nid: str) -> bool:
    if not nid.isdigit():
        return False

    # ถ้า nid ไม่ใช่ 13 ให้คืนค่า False
    if (len(nid) != 13):
        return False

    # บุคคลมี 9 ประเภท 0-8
    if nid.startswith('9'):
        return False

    # ค่าสำหรับอ้างอิง index list ข้อมูลบัตรประชาชน
    num = 0

    # ค่าประจำหลัก
    num2 = 13

    # list ข้อมูลบัตรประชาชน
    listdata = list(nid)

    # ผลลัพธ์
    check_sum = 0
    while num < 12:
        # นำค่า num เป็น  index list แต่ละตัว *  (num2 - num) แล้วรวมเข้ากับ check_sum
        check_sum += int(listdata[num]) * (num2-num)
        # เพิ่มค่า num อีก 1
        num += 1

    # check_sum หาร 11 เอาเศษ
    digit13 = check_sum % 11
    if digit13 == 0:
        # ถ้าเศษ = 0 ค่าหลักที่ 13 คือ 1
        digit13 = 1
    elif digit13 == 1:
        # ถ้าเศษ = 1 ค่าหลักที่ 13 คือ 0
        digit13 = 0
    else:
        # ถ้าเศษไม่ใช่กับอะไร ให้เอา 11 - digit13
        digit13 = 11 - digit13

    if digit13 == int(listdata[12]):
        # ถ้าค่าหลักที่ 13 เท่ากับค่าหลักที่ 13 ที่ป้อนข้อมูลมา คืนค่า True
        return True
    else:
        # ถ้าค่าหลักที่ 13 ไม่เท่ากับค่าหลักที่ 13 ที่ป้อนข้อมูลมา คืนค่า False
        return False
