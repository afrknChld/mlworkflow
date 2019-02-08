import {logger} from "core/logging"
import * as p from "core/properties"
import {empty, label, textarea} from "core/dom"

import {InputWidget, InputWidgetView} from "models/widgets/input_widget"

export class TextAreaView extends InputWidgetView {
  model: TextArea

  protected inputEl: HTMLTextAreaElement

  initialize(options: any): void {
    super.initialize(options)
    this.render()
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.model.change, () => this.render())
  }

  css_classes(): string[] {
    return super.css_classes().concat("bk-widget-form-group")
  }

  render(): void {
    super.render()

    empty(this.el)

    const labelEl = label({for: this.model.id}, this.model.title)
    this.el.appendChild(labelEl)

    this.inputEl = textarea({
      class: "bk-widget-form-input",
      id: this.model.id,
      name: this.model.name,
      disabled: this.model.disabled,
      placeholder: this.model.placeholder,
      style: {minWidth:"90%", minHeight:"100%"},
    }, this.model.value)
    this.inputEl.addEventListener("change", () => this.change_input())
    this.el.appendChild(this.inputEl)

    // TODO - This 35 is a hack we should be able to compute it
    // if (this.model.height)
    //   this.inputEl.style.height = `${this.model.height - 35}px`
  }

  change_input(): void {
    const value = this.inputEl.value
    logger.debug(`widget/textarea: value = ${value}`)
    this.model.value = value
    super.change_input()
  }
}

export namespace TextArea {
  export interface Attrs extends InputWidget.Attrs {
    value: string
    placeholder: string
  }

  export interface Props extends InputWidget.Props {}
}

export interface TextArea extends TextArea.Attrs {}

export class TextArea extends InputWidget {

  properties: TextArea.Props

  constructor(attrs?: Partial<TextArea.Attrs>) {
    super(attrs)
  }

  static initClass(): void {
    this.prototype.type = "TextArea"
    this.prototype.default_view = TextAreaView

    this.define({
      value:       [ p.String, "" ],
      placeholder: [ p.String, "" ],
    })
  }
}

TextArea.initClass()