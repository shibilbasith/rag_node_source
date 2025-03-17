// const { OpenAI } = require("openai");
// const { GoogleGenerativeAI } = require("@google/generative-ai");
const { AutoTokenizer, AutoModel } = require('@huggingface/transformers');
async function loadModel() {
  const tokenizer = await AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct');
  const model = await AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct');
  return { tokenizer, model };
}
async function createEmbedings(text) {
  try {
    // OPEN AI
    // const openai = new OpenAI({
    //   apiKey: process.env.OPENAIKEY,
    // });
    // const embeddings = await openai.embeddings.create({
    //   input: text,
    //   model: "text-embedding-ada-002",
    // });

    //GEMINI AI
    // const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    // const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
    // const embeddings = await embeddingModel.embedContent(text)

    // HUGGING FACE
    const { tokenizer, model } = await loadModel();
    const inputs = tokenizer(text, { return_tensors: 'pt', padding: true, truncation: true });
    const outputs = await model(inputs);
    const embeddings = outputs.last_hidden_state.mean(1).squeeze().tolist(); // Mean pooling

    return embeddings;
  } catch (error) {
    throw new Error(error);
  }
}

module.exports = { createEmbedings };




// const { AutoTokenizer, AutoModel } = require('@huggingface/transformers');

// // Load model and tokenizer
// async function loadModel() {
//   const tokenizer = await AutoTokenizer.from_pretrained('mistralai/mixtral-8x7b-instruct-v0.1');
//   const model = await AutoModel.from_pretrained('mistralai/mixtral-8x7b-instruct-v0.1');
//   return { tokenizer, model };
// }

// // Generate embeddings
// async function getEmbeddings(text) {
//   const { tokenizer, model } = await loadModel();
//   const inputs = tokenizer(text, { return_tensors: 'pt', padding: true, truncation: true });
//   const outputs = await model(inputs);
//   const embeddings = outputs.last_hidden_state.mean(1).squeeze().tolist(); // Mean pooling
//   return embeddings;
// }