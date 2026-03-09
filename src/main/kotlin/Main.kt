package org.example

import ai.djl.ModelException
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ZooModel
import ai.djl.translate.NoopTranslator
import ai.djl.util.Platform
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import kotlin.math.exp
import kotlin.math.max

private const val DEFAULT_MODEL_DIR = "llmlingua2_onnx"
private const val MAX_SEQUENCE_LENGTH = 512
private const val MAX_CONTENT_TOKENS = MAX_SEQUENCE_LENGTH - 2

/**
 * Runs the LLMLingua-2 prompt compressor as a command-line application.
 *
 * Supported options are documented in the built-in usage output.
 *
 * @param args command-line arguments controlling model location and compression behavior
 * @throws IOException if tokenizer native libraries cannot be prepared
 * @throws ModelException if the ONNX model cannot be loaded
 */
@Throws(IOException::class, ModelException::class)
fun main(args: Array<String>) {
    val cli = CliArgs.parse(args)
    val text = cli.text ?: generateSequence(::readLine).joinToString("\n").trim()

    require(text.isNotBlank()) {
        "Provide input text with --text or via stdin."
    }
    require(cli.rate in 0.0..1.0) {
        "--rate must be between 0.0 and 1.0."
    }

    configureTokenizersNativeLibrary()

    Llmlingua2Compressor(cli.modelPath).use { compressor ->
        val result =
            compressor.compress(
                text = text,
                reduceRate = cli.rate,
                forceTokens = cli.forceTokens,
                forceReserveDigits = cli.forceReserveDigits,
            )

        println(result.compressedText)
        System.err.println(
            "Compression ratio: ${formatRatio(result.compressedTokenCount, result.originalTokenCount)} " +
                "(${result.compressedTokenCount}/${result.originalTokenCount} tokens kept)",
        )
    }
}

/**
 * Compresses prompt text with an ONNX-exported LLMLingua-2 model directory.
 *
 * The model directory must contain `model.onnx` and `tokenizer.json`.
 *
 * @param modelPath path to the exported model directory
 */
class Llmlingua2Compressor(
    modelPath: Path,
) : AutoCloseable {
    private val tokenizer = HuggingFaceTokenizer.newInstance(modelPath.resolve("tokenizer.json"))
    private val model: ZooModel<NDList, NDList>
    private val predictor: Predictor<NDList, NDList>

    init {
        configureTokenizersNativeLibrary()

        val criteria =
            Criteria
                .builder()
                .setTypes(NDList::class.java, NDList::class.java)
                .optModelPath(modelPath)
                .optEngine("OnnxRuntime")
                .optTranslator(NoopTranslator())
                .build()

        model = criteria.loadModel()
        predictor = model.newPredictor()
    }

    /**
     * Compresses input text by removing lower-importance tokens.
     *
     * @param text source text to compress
     * @param reduceRate fraction of tokens to remove, from `0.0` to `1.0`
     * @param forceTokens tokens that must always be preserved
     * @param forceReserveDigits whether tokens containing digits must always be preserved
     * @return the compressed text and token counts for the operation
     */
    fun compress(
        text: String,
        reduceRate: Double,
        forceTokens: Set<String>,
        forceReserveDigits: Boolean,
    ): CompressionResult {
        val chunks = chunkText(text)
        if (chunks.isEmpty()) {
            return CompressionResult("", 0, 0)
        }

        val compressedChunks =
            chunks.map {
                compressChunk(
                    chunk = it,
                    reduceRate = reduceRate,
                    forceTokens = forceTokens,
                    forceReserveDigits = forceReserveDigits,
                )
            }

        val compressedText =
            compressedChunks
                .map { it.text.trim() }
                .filter { it.isNotEmpty() }
                .joinToString("\n\n")
                .trim()

        return CompressionResult(
            compressedText = compressedText,
            originalTokenCount = compressedChunks.sumOf { it.originalTokenCount },
            compressedTokenCount = compressedChunks.sumOf { it.compressedTokenCount },
        )
    }

    /** Releases tokenizer and model resources held by this compressor instance. */
    override fun close() {
        predictor.close()
        model.close()
        tokenizer.close()
    }

    private fun compressChunk(
        chunk: String,
        reduceRate: Double,
        forceTokens: Set<String>,
        forceReserveDigits: Boolean,
    ): ChunkCompressionResult {
        val encoding = tokenizer.encode(chunk)
        val ids = stripBoundarySpecialTokens(encoding.ids)
        val tokens = stripBoundarySpecialTokens(encoding.tokens.toList())

        if (ids.isEmpty() || tokens.isEmpty()) {
            return ChunkCompressionResult("", 0, 0)
        }

        val logits = predict(ids)
        val words = aggregateWords(ids, tokens, logits, forceTokens, forceReserveDigits)
        if (words.isEmpty()) {
            return ChunkCompressionResult("", ids.size, 0)
        }

        val threshold = probabilityThreshold(words, reduceRate)
        val keptTokenIds =
            words
                .filter { shouldKeep(it, threshold) }
                .flatMap { it.tokenIds }

        val compressedText =
            if (keptTokenIds.isEmpty()) {
                ""
            } else {
                tokenizer.decode(keptTokenIds.toLongArray()).trim()
            }

        return ChunkCompressionResult(
            text = compressedText,
            originalTokenCount = ids.size,
            compressedTokenCount = keptTokenIds.size,
        )
    }

    private fun predict(ids: LongArray): FloatArray {
        val inputIds = ids.to2d()
        val attentionMask = LongArray(ids.size) { 1L }.to2d()
        val tokenTypeIds = LongArray(ids.size).to2d()

        val manager = model.ndManager.newSubManager()
        try {
            val inputs = NDList()
            inputs.add(manager.create(inputIds).apply { name = "input_ids" })
            inputs.add(manager.create(attentionMask).apply { name = "attention_mask" })
            inputs.add(manager.create(tokenTypeIds).apply { name = "token_type_ids" })

            val outputs = predictor.predict(inputs)
            try {
                return outputs.singletonOrThrow().toFloatArray()
            } finally {
                outputs.close()
                inputs.close()
            }
        } finally {
            manager.close()
        }
    }

    private fun aggregateWords(
        ids: LongArray,
        tokens: List<String>,
        logits: FloatArray,
        forceTokens: Set<String>,
        forceReserveDigits: Boolean,
    ): List<ScoredWord> {
        val words = mutableListOf<ScoredWordBuilder>()

        for (index in tokens.indices) {
            val token = tokens[index]
            if (token in SPECIAL_TOKENS) {
                continue
            }

            val keepProb = keepProbability(logits, index)
            val cleaned = cleanWordPiece(token)
            val startsWord = words.isEmpty() || isWordStart(token) || cleaned in forceTokens

            if (startsWord) {
                words +=
                    ScoredWordBuilder(
                        text = cleaned,
                        tokenIds = mutableListOf(ids[index]),
                        keepProbabilities = mutableListOf(keepProb),
                    )
            } else {
                val current = words.last()
                current.text += cleaned
                current.tokenIds += ids[index]
                current.keepProbabilities += keepProb
            }
        }

        return words.map { builder ->
            val forced = builder.text in forceTokens || (forceReserveDigits && builder.text.any(Char::isDigit))
            val score =
                if (forced) {
                    1.0
                } else {
                    builder.keepProbabilities.maxOrNull() ?: 0.0
                }

            ScoredWord(
                text = builder.text,
                tokenIds = builder.tokenIds.toList(),
                keepProbability = score,
                tokenCount = builder.tokenIds.size,
            )
        }
    }

    private fun probabilityThreshold(
        words: List<ScoredWord>,
        reduceRate: Double,
    ): Double {
        if (reduceRate <= 0.0) {
            return Double.NEGATIVE_INFINITY
        }
        if (reduceRate >= 1.0) {
            return Double.POSITIVE_INFINITY
        }

        val expanded =
            buildList {
                words.forEach { word ->
                    repeat(word.tokenCount) {
                        add(word.keepProbability)
                    }
                }
            }.sorted()

        if (expanded.isEmpty()) {
            return Double.POSITIVE_INFINITY
        }

        val percentile = ((reduceRate * 100.0) + 1.0).coerceAtMost(100.0) / 100.0
        val position = percentile * (expanded.lastIndex)
        val lowerIndex = position.toInt()
        val upperIndex = max(lowerIndex, kotlin.math.ceil(position).toInt())
        if (lowerIndex == upperIndex) {
            return expanded[lowerIndex]
        }

        val fraction = position - lowerIndex
        return expanded[lowerIndex] + (expanded[upperIndex] - expanded[lowerIndex]) * fraction
    }

    private fun shouldKeep(
        word: ScoredWord,
        threshold: Double,
    ): Boolean {
        if (threshold == Double.NEGATIVE_INFINITY) {
            return true
        }
        if (threshold == Double.POSITIVE_INFINITY) {
            return false
        }
        return word.keepProbability > threshold
    }

    private fun chunkText(text: String): List<String> {
        val encoding = tokenizer.encode(text)
        val ids = stripBoundarySpecialTokens(encoding.ids)
        val tokens = stripBoundarySpecialTokens(encoding.tokens.toList())
        if (ids.isEmpty()) {
            return emptyList()
        }

        val chunks = mutableListOf<String>()
        var start = 0
        while (start < ids.size) {
            var endExclusive = minOf(start + MAX_CONTENT_TOKENS, ids.size)
            if (endExclusive < ids.size) {
                val preferred = findChunkBoundary(tokens, start, endExclusive)
                if (preferred > start) {
                    endExclusive = preferred
                }
            }

            val chunkIds = ids.copyOfRange(start, endExclusive)
            val chunkText = tokenizer.decode(chunkIds).trim()
            if (chunkText.isNotEmpty()) {
                chunks += chunkText
            }
            start = endExclusive
        }
        return chunks
    }

    private fun findChunkBoundary(
        tokens: List<String>,
        start: Int,
        endExclusive: Int,
    ): Int {
        for (index in endExclusive - 1 downTo start + 1) {
            if (tokens[index] in CHUNK_END_TOKENS) {
                return index + 1
            }
        }
        return endExclusive
    }
}

/**
 * Result of a prompt compression operation.
 *
 * @property compressedText compressed text reconstructed from the kept tokens
 * @property originalTokenCount number of content tokens before compression
 * @property compressedTokenCount number of content tokens kept after compression
 */
data class CompressionResult(
    val compressedText: String,
    val originalTokenCount: Int,
    val compressedTokenCount: Int,
)

private data class ChunkCompressionResult(
    val text: String,
    val originalTokenCount: Int,
    val compressedTokenCount: Int,
)

private data class ScoredWord(
    val text: String,
    val tokenIds: List<Long>,
    val keepProbability: Double,
    val tokenCount: Int,
)

private data class ScoredWordBuilder(
    var text: String,
    val tokenIds: MutableList<Long>,
    val keepProbabilities: MutableList<Double>,
)

private data class CliArgs(
    val modelPath: Path,
    val rate: Double,
    val text: String?,
    val forceTokens: Set<String>,
    val forceReserveDigits: Boolean,
) {
    companion object {
        fun parse(args: Array<String>): CliArgs {
            var modelPath = Paths.get(System.getProperty("user.home"), DEFAULT_MODEL_DIR)
            var rate = 0.5
            var text: String? = null
            val forceTokens = linkedSetOf("\n", "?", "!", ".", ",")
            var forceReserveDigits = true

            var index = 0
            while (index < args.size) {
                when (val arg = args[index]) {
                    "--model-path" -> {
                        modelPath = Paths.get(requireValue(args, ++index, arg))
                    }

                    "--rate" -> {
                        rate = requireValue(args, ++index, arg).toDouble()
                    }

                    "--text" -> {
                        text = requireValue(args, ++index, arg)
                    }

                    "--force-token" -> {
                        forceTokens += requireValue(args, ++index, arg)
                    }

                    "--no-force-reserve-digits" -> {
                        forceReserveDigits = false
                    }

                    "--help", "-h" -> {
                        printUsageAndExit()
                    }

                    else -> {
                        error("Unknown argument: $arg")
                    }
                }
                index++
            }

            return CliArgs(
                modelPath = modelPath,
                rate = rate,
                text = text,
                forceTokens = forceTokens,
                forceReserveDigits = forceReserveDigits,
            )
        }

        private fun requireValue(
            args: Array<String>,
            index: Int,
            option: String,
        ): String {
            require(index < args.size) { "Missing value for $option" }
            return args[index]
        }
    }
}

private fun keepProbability(
    logits: FloatArray,
    tokenIndex: Int,
): Double {
    val offset = tokenIndex * 2
    val dropLogit = logits[offset].toDouble()
    val keepLogit = logits[offset + 1].toDouble()
    val maxLogit = max(dropLogit, keepLogit)
    val drop = exp(dropLogit - maxLogit)
    val keep = exp(keepLogit - maxLogit)
    return keep / (drop + keep)
}

private fun isWordStart(token: String): Boolean = !token.startsWith("##")

private fun cleanWordPiece(token: String): String = token.removePrefix("##")

private fun LongArray.to2d(): Array<LongArray> = arrayOf(this)

private fun stripBoundarySpecialTokens(ids: LongArray): LongArray {
    if (ids.size >= 2 && ids.first() == 101L && ids.last() == 102L) {
        return ids.copyOfRange(1, ids.size - 1)
    }
    return ids
}

private fun stripBoundarySpecialTokens(tokens: List<String>): List<String> {
    if (tokens.size >= 2 && tokens.first() == "[CLS]" && tokens.last() == "[SEP]") {
        return tokens.subList(1, tokens.size - 1)
    }
    return tokens
}

private fun formatRatio(
    kept: Int,
    original: Int,
): String {
    if (original == 0) {
        return "0.0000"
    }
    return "%.4f".format(kept.toDouble() / original.toDouble())
}

private fun printUsageAndExit(): Nothing {
    println(
        """
        Usage: ./gradlew run --args="--text 'your prompt here' [--rate 0.5]"

        Options:
          --model-path <path>              Path to the LLMLingua-2 ONNX directory.
          --rate <0.0-1.0>                 Fraction of tokens to remove. Default: 0.5
          --text <prompt>                  Prompt to compress. If omitted, stdin is used.
          --force-token <token>            Token that should always be preserved. Repeatable.
          --no-force-reserve-digits        Allow numeric tokens to be removed.
          --help                           Show this message.
        """.trimIndent(),
    )
    kotlin.system.exitProcess(0)
}

private fun configureTokenizersNativeLibrary() {
    val platform = Platform.detectPlatform("tokenizers")
    val libName = System.mapLibraryName("tokenizers")
    val resourcePath = "native/lib/${platform.classifier}/${platform.flavor}/$libName"

    val targetDir = Paths.get("build", "tokenizers-native")
    val targetLib = targetDir.resolve(libName)

    if (!Files.exists(targetLib)) {
        Files.createDirectories(targetDir)
        val loader =
            HuggingFaceTokenizer::class.java.classLoader
                ?: Thread.currentThread().contextClassLoader
        val stream =
            loader.getResourceAsStream(resourcePath)
                ?: throw IllegalStateException("Native tokenizer library not found: $resourcePath")

        stream.use { input ->
            Files.copy(input, targetLib, StandardCopyOption.REPLACE_EXISTING)
        }
    }

    System.setProperty("RUST_LIBRARY_PATH", targetLib.toAbsolutePath().toString())
}

private val SPECIAL_TOKENS = setOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
private val CHUNK_END_TOKENS = setOf(".", "!", "?", ";", ":", ",")
