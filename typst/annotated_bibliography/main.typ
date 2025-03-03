#import "@preview/cmarker:0.1.2"

#cmarker.render(read("german.md"))

#let annotate(key, text) = {
  cite(key, form: "full")

  linebreak()
  linebreak()
  text
  linebreak()
}

#annotate(
  <abs-1906-08253>, list(
    [Introduces technique where they use the dynamics-model in short runs, branching
      frequently from runs using "real life" data. They keep separate replay buffers
      for model-generated data and the real environment data], [It improves the sample efficiency, and avoids overfitting to data in the "real
      data" replay buffer which may have few samples], [They were comparing usefulness of off-policy real data versus on-policy
      model-generated data. They found that branches of 0 length were best in theory,
      but in practice, branches of length 1 were most helpful],
  ),
)

#annotate(
  <WatterSBR15>, list(
    [Model-based learning for images (high dimensional input) that uses latent space
      representation (low dimensional)], [Main contributions: trains the encoder, decoder, and transition model
      simultaneously, which allows them to set a restriction for local linearity for
      the transition dynamics], [The resulting encoder produces good representation spaces. They use KL
      divergence to make sure the model stays in relevant space.],
  ),
)

#annotate(<Williams2004SimpleSG>, "")

#annotate(<MnihKSGAWR13>, "")

#annotate(<Krizhevsky2012ImageNetCW>, "")

#annotate(<SchulmanWDRK17>, "")

#annotate(<SchulmanLMJA15>, "")

#annotate(<IoffeS15>, "")

#annotate(<NIPS1999_464d828b>, "")

#annotate(<Williams2004SimpleSG>, "")

#annotate(<Watkins2004TechnicalNQ>, "")

#annotate(<He2015DeepRL>, "")

#annotate(<Simonyan2014VeryDC>, "")

#annotate(<Krizhevsky2012ImageNetCW>, "")

#bibliography("main.bib")
