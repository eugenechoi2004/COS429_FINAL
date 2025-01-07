import sys
import os
import torch
import argparse
from data_utils import get_data_loaders
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(main_dir, 'models'))
from models import Enc_With_Classifier


def eval_model(loader, enc_path, classifier_path):
    '''
    Evaluate encoder and classifier on test data
    '''

    device_cpu = torch.device("cpu")
    model = Enc_With_Classifier()
    model.encoder.load_state_dict(torch.load(enc_path, map_location = device_cpu, weights_only = True))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location = device_cpu, weights_only = True))
    model.to(device_cpu)
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader['test']:
            images = images.to(device_cpu)
            labels = labels.to(device_cpu)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print('Accuracy: {:.2f}% ({}/{})'.format(accuracy, correct, total))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="source", help="source or ADDA")
    parser.add_argument("--source", help="source domain model is trained on")
    parser.add_argument("--target", help="target domain (if not ADDA, will evaluate on this domain)")
    parser.add_argument("--seed", default="36", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'saved_weights')

    if args.model == "source":
        print(f"Evaluating {args.source} source encoder/classifier on {args.target} test data:")
        eval_model(get_data_loaders(args.target), os.path.join(weights_path, f"src_enc_{args.source}.pt"), os.path.join(weights_path, f"classifier_{args.source}.pt"))
    elif args.model == "ADDA":
        print(f"Evaluating {args.source} -> {args.target} target encoder/classifier on {args.target} test data:")
        eval_model(get_data_loaders(args.target), os.path.join(weights_path, f"target_enc_{args.source}-{args.target}.pt"), os.path.join(weights_path, f"classifier_{args.source}.pt"))

        print(f"Evaluating {args.source} -> {args.target} target encoder/classifier on {args.source} test data:")
        eval_model(get_data_loaders(args.source), os.path.join(weights_path, f"target_enc_{args.source}-{args.target}.pt"), os.path.join(weights_path, f"classifier_{args.source}.pt"))
    else:
        raise Exception("--model must be source or ADDA")
