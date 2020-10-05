r'''
Generate the same masks as Dorylus'
'''

import torch

vtcs = 232965
parts = 60
block_size = vtcs // parts

TRAIN_PORTION = 0.66
VAL_PORTION = 0.1
TEST_PORTION = 0.24

train_mask = torch.zeros(vtcs, dtype=bool)
val_mask   = torch.zeros(vtcs, dtype=bool)
test_mask  = torch.zeros(vtcs, dtype=bool)

for i in range(parts):
    train_stt = i * block_size
    train_end = train_stt + int(block_size * TRAIN_PORTION)
    train_end = min(vtcs, train_end)

    val_stt = train_end
    val_end = val_stt + int(block_size * VAL_PORTION)
    val_end = min(vtcs, val_end)

    test_stt = val_end
    test_end = test_stt + int(block_size * TEST_PORTION)
    test_end = min(vtcs, test_end)

    for j in range(train_stt, train_end):
          train_mask[j] = True
    for j in range(val_stt, val_end):
          val_mask[j] = True
    for j in range(test_stt, test_end):
          test_mask[j] = True

torch.save(train_mask, 'tm.pt')
torch.save(val_mask, 'vm.pt')
torch.save(test_mask, 'sm.pt')
print(train_mask, type(train_mask), train_mask.shape)