# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="Uk3Qw317tZ9Y" outputId="9c2624eb-788d-4b32-8db0-41b5f6bf5030"
print('wassup?')

# + id="lakrD_Em0lyC"
#pip install autoawq

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="TVAd8yem2sod" outputId="ce72e6f5-3866-4915-a893-bbb4eb19d2a8"
# %pip install -U torch torchvision torchaudio

# + id="Rpoazt852syR"


# + colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["7051ac198e25407fae671db3453120aa", "f8c48b2b7d7d450587bde69c7b126a51", "f72cf6602d704e688a5642f869b56d0c", "ef8830c80b72439a92d92be0ee77d6b1", "477a11dec4414c179e3147ce66efc947", "27a2e42f0ed9473a90b120f9e67e7e01", "065937536bdc4873a20d88e94258dea1", "8dbc07340d674a7daf97e1b435c82618", "7ef75bafa9324f2abb1acaf357d61f96", "b0164a57538241888fcd9dd7e29eb64f", "ec05d51d257b4e749baa11677234401d"]} id="3Eoz9F1x0Pva" outputId="70198d67-203c-4e86-8ebb-8bffb31a284f"
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/zephyr-7B-alpha-AWQ"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")

# + colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["7966f9947c3b4ce385fecfc624e8f7b8", "6e19ca9abd7447ca95802c654de8f1c3", "77285e73b9024c1391d90d58be270a10", "c6a64b32a7254c8a9b8bc931a3b36537", "3457a433e7e24bb5855841dcabf62c42", "f231d3eb2bef43f0a5a1d595ab927303", "51a43ecad59945ec8ded5e295fca0c51", "293bad6ed3384dd79a366f87a038bdae", "c97ad29779eb4c0b998f0abbc0ac6b19", "e88f54b3525a40f586548f4127c9b0b1", "0ea21ba37e994e128b48cdf001c6680e", "8d6ff83e510341c596f912b615f5b394", "aa081c9473e74e0d836ad4701be09cea", "7e3aaa2d465142c1ab6e50334c2d3656", "428115067a694f47bb5914db532e2238", "969bd1050844423599c5c19f2c538f29", "3c275ace15ae4d25b9a0dd463be22b60", "08ff3059ccfd4a86991c85544b87749d", "15e356a6250e476c83b5bde047b165f5", "e393851a3a154894816705393b6cb0c8", "1f482e3cef2849fe82809a8743b1ad07", "aae87f72041d4702a354fb6e1ebabf8b", "52b2f0c9f99b451eb25cf7c505d3fd1b", "74ac52ce64ba44ac8fd8a4b48b15d19f", "5a0a170b441b49d389ecddf342bec8d6", "a337ca4db9fc437eabf826aa14c1fe03", "752ca073418b4100af9dafd195d465f3", "ef0dc0860215486b839d16e8db84fb6a", "b46757e4e0af409d804a83d7d610ff78", "71385ff450554bb0b6e2a511d0c16548", "9146d437f46b4cf3914f3799a2429849", "1c78105989e74d29ba7eadc730171e5d", "b3d001edd5794f898bab04391896f58d", "449fc47c683f4a6392873e2670cacd55", "477e4ab0dfbe4f6b899e26a1a6b67f43", "c6a566e9c0824a4987e96706a9f73bbc", "560649126721419ca63f6aaa480505d9", "04357b78ef904ac5b3bfdcb9a956f5c4", "03b3b2903c734a078a73dfaae675c67d", "353533ebef794a5e90e9b4b7b93d922c", "55d0a85189644041abd4c60480ac0ad5", "c4aed8626a4b49d7a9294720aa9788eb", "3a5d0dab011a428fb43e2b00053b554a", "5a64990ae7cf4dbc91d0028b910407c4", "776756d3d98f4607aa8b38a326ac1b31", "876548fa326a4d1ca72754421ed8693c", "55315b5700a541ee8741d55e34066051", "c54d5c9d9e754f6e80605a7e2f8834a9", "728488a007334aa3a799d2eba5590bb1", "8667b7ea06724ec68232a176674ad124", "35d57dd189a64ecda37ec5b62337a430", "a242a6911da34c009e5703eed9493ce0", "d8e88fcbc1744aa0bd9a59676fe70861", "83115a380f2d41078da5a1dfda101ed7", "a3d8dbbe12dc482e82ac8f4761161ab6"]} id="t2Wp-tzZ0Z3A" outputId="68343d5b-ef4d-4e4d-82f5-aaecd04350c2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


# + id="grY6z5DY7R3W"
prompt = "Tell me about AI"
prompt_template='''<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>

'''

# + colab={"base_uri": "https://localhost:8080/"} id="Bp0b3BJW-dTT" outputId="d275fa64-afb4-4669-a77b-596179c3ba41"
generation_output = model.generate(
    tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512
)

print("Output: ", tokenizer.decode(generation_output[0]))


# + id="QXkJqhmF-phD"
def generate(txt):
  tokens = tokenizer(
    prompt_template.format(prompt=txt),
    return_tensors='pt'
  ).input_ids.cuda()

  return tokenizer.decode(
      model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
      )[0]
  )



# + colab={"base_uri": "https://localhost:8080/", "height": 70} id="4kkeuKr6As23" outputId="efbacdc4-4521-4f5f-8538-80e08e960cad"
generate("You are Zephyr-7b-beta, also know as just Zephyr for short. How are you, Zephyr? Please answer as if you were a conscious being.")

# + colab={"base_uri": "https://localhost:8080/"} id="JzL3uDPsAu3W" outputId="a81ccad4-de93-4778-b660-7dc1ccf8500f"
print(generate("""What instrument did John Coltrane play?"""))

# + colab={"base_uri": "https://localhost:8080/"} id="z7SVMUw6Bjze" outputId="8b89e831-0145-4d7c-fea3-2f391bd77a1a"
print(generate("""Who played piano with John Coltrane?"""))

# + colab={"base_uri": "https://localhost:8080/"} id="TdkTq10zCY-x" outputId="b65de52c-e5cf-4575-c14d-1e80cb5f72da"
print(generate("""What is the best Jazz album of all time?"""))

# + colab={"base_uri": "https://localhost:8080/"} id="S4UNd6y4DwmO" outputId="8511baaa-4a97-4a94-8199-af051ec3029f"
print(generate("""Name some of the great jazz bass players."""))

# + colab={"base_uri": "https://localhost:8080/"} id="x-rUeto_EFun" outputId="b14044ee-69b0-41d1-e214-3ce8a3f4d47f"
print(generate("""What notes are in an F minor 11th chord?"""))

# + colab={"base_uri": "https://localhost:8080/"} id="T5UFnbMJIXQD" outputId="e152e9b2-9c3b-48fd-ca90-28bb28cccf74"
print(generate("""Name the classes of medications prescribed for high blood pressure and their modes of action."""))

# + id="qL15C_yJI-eG"

