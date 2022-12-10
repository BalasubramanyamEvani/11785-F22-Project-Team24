import torch
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable


img_norm_means = [0.485, 0.456, 0.406]
img_norm_stds = [0.229, 0.224, 0.225]


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_norm_means, std=img_norm_stds)
])

def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Attack:
    def __init__(self):
        super()
        self.device = None
        self.encoder = None
        self.beam_size = None
        self.decoder = None
        self.index_to_token_dict = None
        self.token_to_index_dict = None

    def load_image(self, image_path):
        image = pil_loader(image_path)
        image = torch.FloatTensor(data_transforms(image))
        image = image.unsqueeze(0)
        image = Variable(image)
        image = image.to(self.device)
        image.requires_grad_()
        return image

    def gen_caption(self, image):
        img_features = self.encoder(image)
        img_features = img_features.expand(self.beam_size, img_features.size(1), img_features.size(2))
        sentence, _, preds = self.decoder.caption(img_features, beam_size=self.beam_size, return_preds=True)
        
        sentence_tokens = []
        for word_idx in sentence:
            sentence_tokens.append(self.index_to_token_dict[str(word_idx)])
            if word_idx == int(self.token_to_index_dict["<eos>"]):
                break
    
        sentence_tokens = sentence_tokens[1:-1]
        sentence = " ".join(sentence_tokens)
        return sentence, preds
    
    def calculate_loss(self, input, target):
        loss = torch.nn.MSELoss()
        return loss(input,target)


class FGSM(Attack):
    def __init__(self, 
        encoder, 
        decoder, 
        token_to_index_dict, 
        index_to_token_dict, 
        trigger_image_path, 
        device, 
        eps=0.02,
        beam_size=3,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.eps = eps
        self.sentence_emb_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.device = device
        self.token_to_index_dict = token_to_index_dict
        self.index_to_token_dict = index_to_token_dict
        self.beam_size = beam_size
        self.img_norm_means = img_norm_means
        self.img_norm_stds = img_norm_stds
        self.trigger_image_path = trigger_image_path

    def perturb(self, image_path):
        self.encoder.eval()
        self.decoder.eval()

        image = self.load_image(image_path)
        self.trigger_image = self.load_image(self.trigger_image_path)
        self.trigger_image.requires_grad = False

        sentence, preds = self.gen_caption(image)
        target_sentence, target_preds = self.gen_caption(self.trigger_image)

        pred_loss = self.calculate_loss(preds, target_preds)
        
        sentence_embedding = torch.Tensor(self.sentence_emb_model.encode(sentence))
        target_sentence_embedding = torch.Tensor(self.sentence_emb_model.encode(target_sentence))
        
        emd_loss = self.calculate_loss(sentence_embedding, target_sentence_embedding)

        pred_loss.backward(emd_loss)

        adversarial_image = image + self.eps * image.grad.sign()
        adversarial_image = torch.clamp(adversarial_image, min=-1, max=1)

        for channel in range(3):
            adversarial_image[0, channel, :, :] = adversarial_image[0, channel, :, :] * self.img_norm_stds[channel] + self.img_norm_means[channel]

        return adversarial_image.detach()


class PGD(Attack):
    def __init__(self, 
        encoder, 
        decoder, 
        token_to_index_dict, 
        index_to_token_dict, 
        trigger_image_path, 
        device, 
        eps=8/255,
        alpha=2/255,
        steps=10,
        beam_size=3,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.sentence_emb_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.device = device
        self.token_to_index_dict = token_to_index_dict
        self.index_to_token_dict = index_to_token_dict
        self.beam_size = beam_size
        self.img_norm_means = img_norm_means
        self.img_norm_stds = img_norm_stds
        self.trigger_image_path = trigger_image_path

    def perturb(self, image_path):
        self.encoder.eval()
        self.decoder.eval()

        image = self.load_image(image_path)

        self.trigger_image = self.load_image(self.trigger_image_path)
        self.trigger_image.requires_grad = False

        curr_image = image

        for _ in range(self.steps):
            sentence, preds = self.gen_caption(curr_image)
            target_sentence, target_preds = self.gen_caption(self.trigger_image)

            pred_loss = self.calculate_loss(preds, target_preds)
            
            sentence_embedding = torch.Tensor(self.sentence_emb_model.encode(sentence))
            target_sentence_embedding = torch.Tensor(self.sentence_emb_model.encode(target_sentence))
            
            emd_loss = self.calculate_loss(sentence_embedding, target_sentence_embedding)

            pred_loss.backward(emd_loss)

            adversarial_image = image + self.alpha * image.grad.sign()
            delta = torch.clamp(adversarial_image - image, min=-self.eps, max=self.eps)
            adversarial_image = torch.clamp(image + delta, min=-1, max=1)
            curr_image = adversarial_image

        for channel in range(3):
            adversarial_image[0, channel, :, :] = adversarial_image[0, channel, :, :] * self.img_norm_stds[channel] + self.img_norm_means[channel]

        return adversarial_image.detach()
