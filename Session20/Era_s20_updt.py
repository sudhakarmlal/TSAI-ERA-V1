import transformers as t
assert t.__version__=='4.25.1', "Transformers version should be as specified"


import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os
import io
import base64
import torch.nn.functional as F
#from pytorch_grad_cam.utils.image import show_cam_on_image


torch.manual_seed(1)

if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

import sys,gc,traceback
import fastcore.all as fc

# %% ../nbs/11_initializing.ipynb 11
def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

# %% ../nbs/11_initializing.ipynb 12
def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

# %% ../nbs/11_initializing.ipynb 13
def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()

clean_mem()

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);

embeds_folder = Path('C:/Users/shivs/Downloads/paintings_embed')
file_names = [path.name for path in embeds_folder.glob('*') if path.is_file()]
print(file_names)

style_names = [list(torch.load(embeds_folder/file).keys())[0] for file in file_names]
style_names
num_added_tokens = tokenizer.add_tokens(style_names)

added_tokens = list(map(tokenizer.added_tokens_encoder.get,style_names))
added_tokens,style_names


text_encoder.resize_token_embeddings(len(tokenizer))
text_encoder.text_model.embeddings.token_embedding


style_dict = {}

list_styles = [torch.load(embeds_folder/file) for file in file_names]


for k,v in list_styles[0].items():
    print(k,v.shape)

style_dict = {style:embedding for each_style in list_styles for style,embedding in each_style.items()}

list(style_dict)

for token,style in zip(added_tokens,style_names):
    text_encoder.text_model.embeddings.token_embedding.weight.data[token] = style_dict[style]

# #checking if we added the embeddings properly to text_encoder
# ft_dict = torch.load(embeds_folder/'fairy-tale-painting_embeds.bin')

# list(ft_dict.keys())[0]

# ft_dict['<fairy-tale-painting-style>'][:10]

clean_mem()

# text_encoder.get_input_embeddings()(torch.tensor(49408, device=torch_device))[:10]


# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# Access the embedding layer
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
token_emb_layer # Vocab size 49408, emb_dim 768

pos_emb_layer = text_encoder.text_model.embeddings.position_embedding

position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)
print(position_embeddings.shape)

def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output

#Generating an image with these modified embeddings

def generate_with_embs_custom(text_embeddings,seed):
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 1            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1

    max_length = text_embeddings.shape[1]
    uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)[0]


# ref_image = Image.open('C:/Users/shivs/Downloads/lg.jpg').resize((512,512))
# ref_latent = pil_to_latent(ref_image)

## Guidance through Custom Loss Function
def custom_loss(latent):
    error = F.mse_loss(0.5*latent,0.8*ref_latent)
    return error


class Styles_paintings():
    def __init__(self,prompt):
        self.output_styles = []
        self.prompt = prompt
        self.style_names =  list(style_dict)
        self.seeds = [1024+i  for i in range(len(self.style_names))]
        
    def generate_styles(self): 
        #print('The Values are ', list(style_dict)[0])
        
        for seed,style_name  in zip(self.seeds,self.style_names):
            # Tokenize
            prompt = f'{self.prompt} in the style of {style_name}'
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,  return_tensors="pt")
            input_ids = text_input.input_ids.to(torch_device)

            # Get token embeddings
            token_embeddings = token_emb_layer(input_ids)


            # Combine with pos embs
            input_embeddings = token_embeddings + position_embeddings

            #  Feed through to get final output embs
            modified_output_embeddings = get_output_embeds(input_embeddings)

            # And generate an image with this:
            self.output_styles.append(generate_with_embs_custom(modified_output_embeddings,seed))
        
    def generate_styles_with_custom_loss(self, image):
        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = 1  #@param           # Number of denoising steps
        guidance_scale = 8 #@param               # Scale for classifier-free guidance
        batch_size = 1
        custom_loss_scale = 200 #@param
        #print('image shape there is',image.size)
        self.output_styles_with_custom_loss = []
        #ref_image = Image.open('C:/Users/shivs/Downloads/ig.jpg').resize((512,512))
        ref_latent = pil_to_latent(ref_image)            
        for seed,style_name  in zip(self.seeds,self.style_names):
                # Tokenize
            prompt = f'{self.prompt} in the style of {style_name}'
            generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
            print(f' the prompt is :  {prompt} with seed value :{seed}')  
            # Prep text
            text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

            # And the uncond. input as before:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            with torch.no_grad():
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Prep Scheduler
            set_timesteps(scheduler, num_inference_steps)

            # Prep latents
            latents = torch.randn(
              (batch_size, unet.in_channels, height // 8, width // 8),
              generator=generator,)
            latents = latents.to(torch_device)
            latents = latents * scheduler.init_noise_sigma

            # Loop
            for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                #### ADDITIONAL GUIDANCE ###
                if i%5 == 0:
                    # Requires grad on the latents
                    latents = latents.detach().requires_grad_()

                    # Get the predicted x0:
                    latents_x0 = latents - sigma * noise_pred
                    #latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

                    # Decode to image space
                    #denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

                    # Calculate loss
                    loss = custom_loss(latents_x0) * custom_loss_scale
                    #loss = blue_loss(denoised_images) * blue_loss_scale

                    # Occasionally print it out
                    if i%10==0:
                        print(i, 'loss:', loss.item())

                    # Get gradient
                    cond_grad = torch.autograd.grad(loss, latents)[0]

                    # Modify the latents based on this gradient
                    latents = latents.detach() - cond_grad * sigma**2

                # Now step with scheduler
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            self.output_styles_with_custom_loss.append(latents_to_pil(latents)[0])

def generate_final_image(im1,in_prompt):
    paintings = Styles_paintings(in_prompt)
    paintings.generate_styles()
    r_image = im1.resize((512,512))
    print('image shape is',r_image.size)
    paintings.generate_styles_with_custom_loss(r_image)
    
    #print(len(paintings.output_styles))

    return [paintings.output_styles[0]], [paintings.output_styles[1]],[paintings.output_styles[2]],[paintings.output_styles[3]],[paintings.output_styles[4]],[paintings.output_styles_with_custom_loss[0]],[paintings.output_styles_with_custom_loss[1]],[paintings.output_styles_with_custom_loss[2]],[paintings.output_styles_with_custom_loss[3]],[paintings.output_styles_with_custom_loss[4]]
