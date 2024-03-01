const { OpenAI } = require("openai");
async function createEmbedings(text) {
  try {
    const openai = new OpenAI({
      apiKey: process.env.OPENAIKEY,
    });

    const embeddings = await openai.embeddings.create({
      input: text,
      model: "text-embedding-ada-002",
    });

    return embeddings;
  } catch (error) {
    throw new Error(error);
  }
}

module.exports = { createEmbedings };
