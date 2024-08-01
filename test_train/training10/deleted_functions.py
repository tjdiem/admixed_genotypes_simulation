            # prediction_bases = torch.tensor([0.25, 0.5, 0.25]).to(device)
            # tmp = ((out_smoothed - prediction_bases) / (out - prediction_bases).unsqueeze(1)).mean(dim=-1, keepdim=True)
            # tmp = tmp ** 2

            # print(ind3)
            # print(ind4)

            # idx1, idx2 = 13, 0
            # a = predictions[ind3][idx1, input_size // 2 + idx2]
            # b = out_smoothed[idx1, input_size // 2 + idx2]
            # c = tmp[idx1, input_size // 2 + idx2]
            # print(a) ; print(b) ; print(c) ; print()
            # print(a.sum().item()) ; print(b.sum().item()) ; print()
            # print(out[idx1])

            # if torch.rand((1,)).item() < 0.01:
            #     print(ind3)
            #     print(ind4)
            #     print(tmp[2, input_size // 2: -input_size // 2])
            #     print(out_smoothed[2, input_size // 2: -input_size // 2])
            #     print(predictions[ind3][2, input_size // 2: -input_size // 2])
            #     print()
            #     exit()

            if not (tmp >= 0).all():
                print(i)
                print(tmp)
                print(tmp.min())
                assert False
            if not (tmp <= 1).all():
                print(i)
                print(tmp)
                print(tmp.max())
                assert False

            if not ((out_smoothed[:, input_size // 2 : - (input_size // 2)].sum(dim=-1) - 1).abs() < 1e-4).all():
                print(out_smoothed[:, input_size // 2 : - (input_size // 2)].sum(dim=-1)[0])
                print(out_smoothed[:, input_size // 2 : - (input_size // 2)].sum(dim=-1))
                print(i)
                print((out_smoothed[:, input_size // 2 : - (input_size // 2)].sum(dim=-1) - 1).abs().max())
                for k in range(batch_size):
                    for j in range(len_seq):
                        if abs(out_smoothed[k, input_size // 2 + j].sum().item() - 1) >= 1e-4:
                            print(k, j)
                            print(out_smoothed[k, input_size // 2 + j])
                assert False




            # print(a * (1 - c) + b * c)

            # print("out smoothed")
            # for k in range(batch_size):
            #     for j in range(len_seq):
            #         if abs(out_smoothed[k, input_size // 2 + j].sum().item() - 1) > 1e-2:
            #             print(k, j)
            #             print(out_smoothed[k, input_size // 2 + j])

            # print("predictions")
            # for k in range(num_individuals):
            #     for j in range(len_seq):
            #         if abs(predictions[k, input_size // 2 + j].sum().item() - 1) > 1e-4:
            #             print(ind3)
            #             print(ind4)
            #             print(i)
            #             print(k, j)
            #             print(predictions[k, input_size // 2 + j])
            #             print(predictions[k, input_size // 2 + j].sum() - 1.0)
            #             exit()

            # print(((1 - predictions[:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs()).max())
            # print(((1 - predictions[:, input_size // 2: - (input_size // 2)].sum(dim=-1)).abs()).argmax())


            
            # idx = 3
            # print(transitions[idx, ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2, 0,0])
            # print(ind3)
            # print(ind4[idx])
            # print(out[idx])
            # print(tmp[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,0])
            # print(tmp[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,1])
            # print(tmp[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,2])
            # print(out_smoothed[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,0]) # batch, num_positions, num_classes
            # print(out_smoothed[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,1])
            # print(out_smoothed[idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,2])
            # print(predictions[ind3][idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,0])
            # print(predictions[ind3][idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,1])
            # print(predictions[ind3][idx,ind4[idx] - 100 + input_size // 2: ind4[idx] + 100 + input_size // 2,2])
            # exit()