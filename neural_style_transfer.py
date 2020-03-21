from utils import *

model = models.vgg19(pretrained=True).features
# setting everything to eval mode
for parameter in model.parameters():
    parameter.requires_grad = False
model.to(device)

transform = transforms.Compose([transforms.Resize(300),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# content image
content = Image.open("content.jpg").convert("RGB")
# plt.imshow(content)
# plt.show()
content = transform(content).to(device)
# initial generated image
# noise = generate_noise(content).to(device)
generated_image = content.clone()  # + noise.clone()
generated_image = generated_image.clone().requires_grad_(True).to(device)
# plt.imshow(generated_image.cpu().detach().numpy().transpose(1, 2, 0))

# style image
style = Image.open("style.jpg").convert("RGB")
# plt.imshow(style)
# plt.show()
style = transform(style).to(device)

content_activations = get_content_activations(content, model, layer_name=['21'])

# layers from where we want to get style
layer_names = ['0', '5', '10', '19', '28']

# weights to apply to each of the layers
style_weights = {
    "0": 0.2,
    "5": 0.2,
    "10": 0.2,
    "19": 0.2,
    "28": 0.2
}
style_grams = get_style_grams(style, model, layer_names)
optimizer = torch.optim.Adam([generated_image], lr=0.1)

num_epochs = 50
for i in range(1, num_epochs + 1):
    total_loss = get_loss(x=generated_image,
                          model=model,
                          weights=style_weights,
                          style_grams=style_grams,
                          content_acts=content_activations,
                          content_layer_name=['21'],
                          alpha=10,
                          beta=40)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # if i % 10 == 0:
    #   plt.imshow(imcnvt(generated_image))
    #   plt.show()

plt.imshow(imcnvt(generated_image))
plt.show()
