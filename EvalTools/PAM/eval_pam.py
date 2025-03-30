import argparse
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from PAM import PAM
from dataset import ExampleDatasetFolder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PAM")
    # parser.add_argument('--folder', type=str, help='Folder path to evaluate')
    parser.add_argument('--wavscp', type=str, help='Folder path to evaluate')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples per batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    args = parser.parse_args()

    f_out = open(args.wavscp + ".pam_score.json",'w')
    proc_file = {}
    with open(args.wavscp) as f:
        for line in f:
            utt = line.strip()
            input_file = json.loads(utt)
            ref_wav = input_file["location"]
            proc_file[ref_wav] = input_file

    # initialize PAM
    model_path = "./saved/pretrained_model/clap/msclap/msclap/CLAP_weights_2023.pth"
    pam = PAM(model_fp=model_path, use_cuda=torch.cuda.is_available())

    # Create Dataset and Dataloader
    dataset = ExampleDatasetFolder(
        src=args.wavscp,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False,
                            num_workers = args.num_workers,
                            pin_memory = False, drop_last=False, collate_fn=dataset.collate)

    # Evaluate and print PAM score
    collect_pam, collect_pam_segment = [], []
    files_score = {}
    for files, audios, sample_index in tqdm(dataloader):
        pam_score, pam_segment_score = pam.evaluate(audios, sample_index)
        collect_pam += pam_score
        collect_pam_segment += pam_segment_score

        #print("files", files, pam_score, pam_segment_score)
        for file_audio, pam_score_audio in zip(files, pam_score):
            proc_file[file_audio]["q_score"] = pam_score_audio
            input_file_json = json.dumps(proc_file[file_audio], ensure_ascii=False)
            f_out.write(input_file_json+"\n")
            f_out.flush()

    mean_pam = sum(collect_pam)/len(collect_pam)
    print(f"PAM Score: {sum(collect_pam)/len(collect_pam)}")
    with open(args.wavscp + ".PAM_score.mean",'w') as f:
        f.write(str(mean_pam))
