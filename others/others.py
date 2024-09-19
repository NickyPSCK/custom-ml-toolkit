def create_bin(series, bins):
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
