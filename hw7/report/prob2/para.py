from architecuture import StudentNet

if __name__ == '__main__':
    model = StudentNet()
    count = sum(p.numel() for p in model.parameters())
    print('# of model parameters:', count)