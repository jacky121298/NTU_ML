from model import StudentNet

if __name__ == '__main__':
    model = StudentNet()
    count = sum(p.numel() for p in model.parameters())
    print('# of model parameters:', count)

    model = StudentNet(width_mult = 0.23)
    count = sum(p.numel() for p in model.parameters())
    print('# of pruning model parameters:', count)