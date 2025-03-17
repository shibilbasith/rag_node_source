var express = require("express");
var router = express.Router();
const { MongoClient, ObjectId } = require("mongodb");
const { createEmbedings } = require("./embedings");
const { OpenAI } = require("openai")
const fs = require("fs");

var PDFParser = require("pdf2json");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const parser = new PDFParser(this, 1);

/* GET home page. */
router.get("/", async function (req, res, next) {
  try {
    const connection = await MongoClient.connect(process.env.DB);
    const db = connection.db("rag_doc");
    const collection = db.collection("docs");
    await collection.insertOne({ test: "Success" });
    await connection.close();
    res.json({ title: "Express" });
  } catch (error) {
    console.log(error);
  }
});

router.post("/load-document", async (req, res) => {
  try {
    parser.loadPDF("./docs/policy.pdf");
    parser.on("pdfParser_dataReady", async (data) => {
      await fs.writeFileSync("./context.txt", parser.getRawTextContent());

      const content = await fs.readFileSync("./context.txt", "utf-8");
      const splitContent = content.split("\n");

      const connection = await MongoClient.connect(process.env.DB);
      const db = connection.db("rag_doc");
      const collection = db.collection("docs");

      for (line of splitContent) {
        const embedings = await createEmbedings(line);
        await collection.insertOne({
          text: line,
          // embedding: embedings.data[0].embedding,
          embedding: embedings.embedding.values,
        });
        console.log(line);
      }
      await connection.close();
      res.json("Done");
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: "Error" });
  }
});

router.get("/embeddings", async (req, res) => {
  try {
    const embedings = await createEmbedings("Hello World");
    console.log("🚀 ~ router.get ~ embedings:", embedings)
    res.json(embedings);
  } catch (error) {
    console.log(error);
    res.status(500).json({ meesage: "Error" });
  }
});

router.post("/conversation", async (req, res) => {
  try {
    let sessionId = req.body.sessionId;
    const connection = await MongoClient.connect(process.env.DB);
    const db = connection.db("rag_doc");

    if (!sessionId) {
      const collection = db.collection("sessions");
      const sessionData = await collection.insertOne({ createdAt: new Date() });
      sessionId = sessionData._id;
    }

    if (sessionId) {
      const collection = db.collection("sessions");
      const sessionData = await collection.findOne({
        _id: new ObjectId(sessionId),
      });
      if (sessionData) {
        sessionId = sessionData._id;
      } else {
        return res.json({
          message: "Session Not Found",
        });
      }
    }

    // Lets work conversation
    const message = req.body.message;
    const conCollection = db.collection("conversation");
    await conCollection.insertOne({
      sessionId: sessionId,
      message: message,
      role: "USER",
      createdAt: new Date(),
    });

    // Convert message to vector
    console.log(req.body.message);
    const embedings = await createEmbedings(req.body.message);

    const docsCollection = db.collection("docs");
    const vectorSearch = await docsCollection.aggregate([
      {
        $vectorSearch: {
          index: "default",
          path: "embedding",
          // queryVector: embedings.data[0].embedding,
          queryVector: embedings.embedding.values,
          numCandidates: 150,
          limit: 10,
        },
      },
      {
        $project: {
          _id: 0,
          text: 1,
          score: {
            $meta: "vectorSearchScore",
          },
        },
      },
    ]);

    let finalResult = []

    for await (let doc of vectorSearch) {
      finalResult.push(doc)
    }

    // const ai = new OpenAI({
    //   apiKey : process.env.OPENAIKEY
    // })

    // const chat = await ai.chat.completions.create({
    //   model : "gpt-4",
    //   messages : [
    //     {
    //       role : "system",
    //       content : "You are a humble helper who can answer for questions asked by users from the given context."
    //     },
    //     {
    //       role : "user",
    //       content : `${finalResult.map(doc => doc.text + "\n")}
    //       \n
    //       From the above context, answer the following question: ${message}`
    //     }
    //   ]
    // })
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const generativeModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    // const prompt = `Query: ${message}\n\nContext: ${finalResult.map(doc => doc.text + "\n")}\n\nAnswer the query using the provided context.`;
    const prompt = `Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Use three sentences maximum and keep the answer as concise as possible.
                    Always say "thanks for asking!" at the end of the answer.

                    ${finalResult.map(doc => doc.text + "\n")}

                    Question: ${message}

                    Helpful Answer:`;
                    
    const result = await generativeModel.generateContent(prompt);
    console.log("🚀 ~ router.post ~ result:", result.response.candidates[0].content.parts[0].text)
    // return result.response.text();

    console.log(`${finalResult.map(doc => doc.text + "\n")}
    \n
    From the above context, answer the following question: ${message}`)

    // return res.json(chat.choices[0].message.content);
    return res.json(result.response.candidates[0].content.parts[0].text);
  } catch (error) {
    res.json({ message: "Something went wrong" });
    console.log(error);
  }
});

module.exports = router;
