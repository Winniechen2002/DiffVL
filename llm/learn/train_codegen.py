import argparse
import os
import transformers
from datasets import load_dataset
from llm import LLMPATH
from transformers.testing_utils import CaptureLogger
from llm.learn.codegpt import CodeNet
from transformers import Trainer, TrainingArguments


def load_my_dataset():
    from llm.pl.tester.integer import int_dsl as dsl
    return dsl, load_dataset(os.path.join(LLMPATH, 'learn/dataset/simple'), cache_dir=None)
         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    args = parser.parse_args()

    dsl, raw_dataset = load_my_dataset()
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    model = CodeNet.from_dsl(dsl)
    tokenizer = model.tokenizer

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples)
        return output

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=None,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    import evaluate
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels, inputs = eval_preds
        labels = inputs
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)

        """
        from llm.pl.tester.integer import incr, test
        from llm.learn.serialize import extract_code
        inputs = model.tokenizer.batchify([extract_code(test)])
        cr = model.tokenizer.token2id['<CR>']
        print(inputs['input_ids'])
        p = model(**{k: v for k, v in inputs.items()}).logits[0].argmax(axis=-1)
        print(model.tokenizer.decode(p))
        print(p)
        """

        return metric.compute(predictions=preds, references=labels)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)


    train = tokenized_dataset['train']
    valid = tokenized_dataset['validation']
    def data_collator(features):
        return tokenizer.batchify(features, should_map=False, device='cpu')

    training_args = TrainingArguments(output_dir='tmp_trainer', evaluation_strategy="epoch", num_train_epochs=args.num_train_epochs, include_inputs_for_metrics=True)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid, #NOTE: eval on the train set ..
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    


    # TODO: detect last checkpoint
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    #metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':
   main()